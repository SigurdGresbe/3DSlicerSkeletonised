import json
import logging
from pathlib import Path
from typing import Any, Dict

from PyQt5.QtGui import QColor

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)

USER_SETTINGS_PATH = PACKAGE_ROOT / "user_settings.json"
DEFAULTS_PATH = PACKAGE_ROOT / "config_defaults.json"

class SettingsManager:
    """
    Manages loading and saving user settings from JSON files.
    - Loads defaults from config_defaults.json.
    - Loads user settings from user_settings.json.
    - Merges user settings over defaults.
    """
    def __init__(self):
        self.settings: Dict[str, Any] = {}
        self.load()

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Safely loads a JSON file."""
        if not path.is_file():
            logger.debug(f"Settings file not found: {path}")
            return {}
        try:
            with open(path, 'r', encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load settings from {path}: {e}")
            return {}

    def _save_json(self, path: Path, data: Dict[str, Any]):
        """Safely saves data to a JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info(f"User settings saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save settings to {path}: {e}")

    def load(self):
        """Loads defaults, then merges user settings on top."""
        defaults = self._load_json(DEFAULTS_PATH)
        if not defaults:
            logger.error("FATAL: Could not load config_defaults.json. Settings will be empty.")
            
        user = self._load_json(USER_SETTINGS_PATH)

        self.settings = defaults
        self.settings.update(user)
        logger.debug("Settings loaded and merged.")

    def save(self):
        """Saves the current settings to the user file."""

        defaults = self._load_json(DEFAULTS_PATH)
        user_settings = {}
        for key, value in self.settings.items():
            if key not in defaults or defaults[key] != value:
                user_settings[key] = value
                
        self._save_json(USER_SETTINGS_PATH, user_settings)

    def save_as_defaults(self, settings=None, exclude_keys=None) -> bool:
        """Saves the current settings as the new default configuration."""
        exclude_keys = set(exclude_keys or [])
        source_settings = self.settings if settings is None else dict(settings)
        defaults = {
            key: value
            for key, value in source_settings.items()
            if key not in exclude_keys
        }

        try:
            DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(defaults, f, indent=4)

            if USER_SETTINGS_PATH.is_file():
                USER_SETTINGS_PATH.unlink()

            self.settings = defaults
            logger.info("Current settings saved as defaults to %s", DEFAULTS_PATH)
            return True
        except Exception as e:
            logger.error("Failed to save current settings as defaults: %s", e)
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a setting by key."""
        return self.settings.get(key, default)

    def set(self, key: str, value: Any):
        """Sets a setting by key."""
        self.settings[key] = value

    def get_as_qcolor(self, key: str) -> QColor:
        """Helper to get a setting as a QColor object."""
        return QColor(self.settings.get(key, "#ffffff"))

    def set_from_qcolor(self, key: str, color: QColor):
        """Helper to set a QColor object as a hex string."""
        self.settings[key] = color.name()

    def reset_to_defaults(self):
        """Deletes the user settings file and reloads the defaults."""
        if USER_SETTINGS_PATH.is_file():
            try:
                USER_SETTINGS_PATH.unlink()
                logger.info("User settings file deleted. Reverting to defaults.")
            except Exception as e:
                logger.error(f"Could not delete user settings file: {e}")

        self.load()
