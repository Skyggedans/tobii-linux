# Build and install the Tobii Linux stack.
#
#   make build            # cargo build --release
#   make install          # binaries + libtobii.so + udev rule (sudo) + user units
#   make enable           # start the always-on user service now
#   make uninstall        # remove everything installed by `make install`
#
# Run as your normal user; the system parts invoke sudo themselves. Override
# paths with e.g. `make install PREFIX=/usr`.

PREFIX      ?= /usr/local
BINDIR      ?= $(PREFIX)/bin
LIBDIR      ?= $(PREFIX)/lib
UDEVDIR     ?= /etc/udev/rules.d
USERUNITDIR ?= $(HOME)/.config/systemd/user
SUDO        ?= sudo

REL         := target/release
BINS        := tobiid tobii-opentrack tobii-gaze-keys tobii5-init-replay
LIB         := libtobii.so

.PHONY: build install install-bin install-udev install-units enable enable-keys disable uninstall clean

build:
	cargo build --release

install: build install-bin install-udev install-units
	@echo
	@echo "Installed. Start the daemon with:  make enable"
	@echo "(or socket activation:  systemctl --user enable --now tobiid.socket)"

install-bin:
	@if [ "$$(id -u)" = 0 ]; then echo "run as your user, not root (system parts use sudo)"; exit 1; fi
	$(SUDO) install -d $(BINDIR) $(LIBDIR)
	$(SUDO) install -m 0755 $(addprefix $(REL)/,$(BINS)) $(BINDIR)/
	$(SUDO) install -m 0644 $(REL)/$(LIB) $(LIBDIR)/
	$(SUDO) ldconfig || true

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
	$(SUDO) rm -f $(addprefix $(BINDIR)/,$(BINS)) $(LIBDIR)/$(LIB)
	$(SUDO) rm -f $(UDEVDIR)/99-tobii-uaccess.rules $(UDEVDIR)/99-tobii-no-uvcvideo.rules $(UDEVDIR)/99-tobii-uinput.rules
	$(SUDO) udevadm control --reload || true
	$(SUDO) ldconfig || true

clean:
	cargo clean
