# Build and install the Tobii Linux stack.
#
#   make build            # cargo build --release
#   make install          # binaries + libtobii.so + headers + udev rule (sudo) + user units
#   make enable           # start the always-on user service now
#   make uninstall        # remove everything installed by `make install`
#
# Run as your normal user; the system parts invoke sudo themselves. Override
# paths with e.g. `make install PREFIX=/usr`.

PREFIX      ?= /usr/local
BINDIR      ?= $(PREFIX)/bin
LIBDIR      ?= $(PREFIX)/lib
INCLUDEDIR  ?= $(PREFIX)/include
UDEVDIR     ?= /etc/udev/rules.d
USERUNITDIR ?= $(HOME)/.config/systemd/user
SUDO        ?= sudo

REL         := target/release
# tobii5-init-replay is built but NOT installed: every one of its subcommands
# is log analysis, UVC-camera work or diagnostics, none of which the driver
# needs. Run it from $(REL). It is still removed by `uninstall` for anyone who
# installed it with an older Makefile.
BINS        := tobiid tobii-opentrack tobii-gaze-keys tobii-calibrate
LEGACY_BINS := tobii5-init-replay
LIB         := libtobii.so
INCSRC      := crates/tobii-ffi/include
ABI_SMOKE   := crates/tobii-ffi/abi-smoke.c

# The entry points libtobii.so presents to C — every export of the Stream
# Engine 4.1.0.3 DLL plus tobii_recenter — and the headers that declare them
# are its whole contract with consumers such as OpenTrack's tracker-tobii.
# `verify-abi` checks both against the built library.
ABI_LIST    := crates/tobii-ffi/abi-symbols.txt
ABI_SYMBOLS := $(shell sed 's/ *\#.*//; /^$$/d' $(ABI_LIST))
HEADERS     := $(wildcard $(INCSRC)/tobii/*.h)

.PHONY: build check verify-abi install install-bin install-headers install-udev install-units enable enable-keys disable uninstall clean

build:
	cargo build --release --workspace

# Everything CI would run.
check:
	cargo fmt --all --check
	cargo clippy --release --all-targets --workspace -- -D warnings
	cargo test --workspace
	cargo doc --no-deps --workspace

# libtobii.so must keep exporting exactly the listed entry points, and a
# Stream-Engine-shaped C program must still compile against the headers, link
# against the library and get the answers they promise. abi-smoke takes the
# address of every listed symbol through the headers (so each must be declared
# with C linkage and exported) and needs neither the daemon nor a tracker.
verify-abi: build
	@nm -D --defined-only $(REL)/$(LIB) \
		| awk '$$2 == "T" && $$3 ~ /^tobii_/ { print $$3 }' | sort > $(REL)/abi-actual.txt
	@printf '%s\n' $(ABI_SYMBOLS) | sort > $(REL)/abi-expected.txt
	@diff -u $(REL)/abi-expected.txt $(REL)/abi-actual.txt \
		|| { echo "libtobii.so no longer exports exactly $(ABI_LIST)"; exit 1; }
	@echo "libtobii.so exports all $(words $(ABI_SYMBOLS)) listed tobii_* symbols"
	@printf 'X(%s)\n' $(ABI_SYMBOLS) > $(REL)/abi-symbols.inc
	@$(CC) -std=c11 -Wall -Wextra -Werror -I$(INCSRC) -I$(REL) $(ABI_SMOKE) \
		-L$(REL) -ltobii -Wl,-rpath,$(abspath $(REL)) -lm -o $(REL)/abi-smoke
	@$(REL)/abi-smoke

install: build install-bin install-headers install-udev install-units
	@echo
	@echo "Installed. Start the daemon with:  make enable"
	@echo "(or socket activation:  systemctl --user enable --now tobiid.socket)"

install-bin:
	@if [ "$$(id -u)" = 0 ]; then echo "run as your user, not root (system parts use sudo)"; exit 1; fi
	$(SUDO) install -d $(BINDIR) $(LIBDIR)
	$(SUDO) install -m 0755 $(addprefix $(REL)/,$(BINS)) $(BINDIR)/
	$(SUDO) install -m 0644 $(REL)/$(LIB) $(LIBDIR)/
	$(SUDO) ldconfig || true

# The C headers a Stream Engine client compiles against, e.g.
#   cmake .. -DSDK_TOBII=$(PREFIX)   in OpenTrack's tracker-tobii
install-headers:
	$(SUDO) install -d $(INCLUDEDIR)/tobii
	$(SUDO) install -m 0644 $(HEADERS) $(INCLUDEDIR)/tobii/

install-udev:
	$(SUDO) install -m 0644 systemd/99-tobii-uaccess.rules $(UDEVDIR)/
	$(SUDO) install -m 0644 systemd/99-tobii-no-uvcvideo.rules $(UDEVDIR)/
	$(SUDO) install -m 0644 systemd/99-tobii-uinput.rules $(UDEVDIR)/
	$(SUDO) udevadm control --reload
	$(SUDO) udevadm trigger
	@echo "udev rules installed — re-plug the tracker if it was already connected"

# Install user units, rewriting ExecStart to the installed binary path.
install-units:
	mkdir -p $(USERUNITDIR)
	sed 's|^ExecStart=.*|ExecStart=$(BINDIR)/tobiid|' systemd/tobiid.service \
		> $(USERUNITDIR)/tobiid.service
	install -m 0644 systemd/tobiid.socket $(USERUNITDIR)/tobiid.socket
	sed 's|^ExecStart=%h/Work/tobii/target/release/tobii-gaze-keys|ExecStart=$(BINDIR)/tobii-gaze-keys|' \
		systemd/tobii-gaze-keys.service > $(USERUNITDIR)/tobii-gaze-keys.service
	systemctl --user daemon-reload

enable:
	systemctl --user enable --now tobiid.service

# Enable the gaze-to-keyboard binder too (pulls in tobiid via Requires=).
enable-keys:
	systemctl --user enable --now tobii-gaze-keys.service

disable:
	-systemctl --user disable --now tobii-gaze-keys.service
	-systemctl --user disable --now tobiid.service
	-systemctl --user disable --now tobiid.socket

uninstall: disable
	-rm -f $(USERUNITDIR)/tobiid.service $(USERUNITDIR)/tobiid.socket $(USERUNITDIR)/tobii-gaze-keys.service
	systemctl --user daemon-reload
	$(SUDO) rm -f $(addprefix $(BINDIR)/,$(BINS) $(LEGACY_BINS)) $(LIBDIR)/$(LIB)
	$(SUDO) rm -f $(addprefix $(INCLUDEDIR)/tobii/,$(notdir $(HEADERS)))
	-$(SUDO) rmdir $(INCLUDEDIR)/tobii 2>/dev/null
	$(SUDO) rm -f $(UDEVDIR)/99-tobii-uaccess.rules $(UDEVDIR)/99-tobii-no-uvcvideo.rules $(UDEVDIR)/99-tobii-uinput.rules
	$(SUDO) udevadm control --reload || true
	$(SUDO) ldconfig || true

clean:
	cargo clean
