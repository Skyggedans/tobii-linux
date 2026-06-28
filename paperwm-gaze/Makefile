# tobii gaze switcher — self-contained install into the PaperWM extension.
PAPERWM_EXT ?= $(HOME)/.local/share/gnome-shell/extensions/paperwm@paperwm.github.com

.PHONY: install uninstall test
install:
	@test -d "$(PAPERWM_EXT)" || { echo "PaperWM not found at $(PAPERWM_EXT)"; exit 1; }
	cp gaze.js gazelib.js "$(PAPERWM_EXT)/"
	@grep -q '// tobii-gaze' "$(PAPERWM_EXT)/extension.js" || { \
	  sed -i "/} from '.\/imports.js';/a import * as Gaze from './gaze.js'; // tobii-gaze" "$(PAPERWM_EXT)/extension.js"; \
	  sed -i "/modules = \[/a\\        Gaze, // tobii-gaze" "$(PAPERWM_EXT)/extension.js"; \
	  echo "patched extension.js"; \
	}
	@echo "Installed. Log out and back in (Wayland) to reload PaperWM."

uninstall:
	-sed -i '/\/\/ tobii-gaze/d' "$(PAPERWM_EXT)/extension.js"
	-rm -f "$(PAPERWM_EXT)/gaze.js" "$(PAPERWM_EXT)/gazelib.js"
	@echo "Removed tobii gaze switcher. Log out and back in to reload PaperWM."

test:
	gjs -m tests/test_gazelib.js
