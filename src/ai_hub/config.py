import json
import os
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG = {
    "api": {"host": "0.0.0.0", "port": 8000, "reload": False},
    "model": {
        "default_input_dim": 128,
        "default_hidden_dim": 256,
        "default_output_dim": 10,
        "default_epochs": 10,
        "models_dir": "models",
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "ai_hub.log",
    },
}


class Config:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config = self._load_config()

    def _get_default_config_path(self) -> str:
        return str(Path.home() / ".ai_hub" / "config.json")

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        return DEFAULT_CONFIG.copy()

    def save(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self._config, f, indent=2)

    def get(self, key: str, default=None):
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save()


# Global config instance
config = Config()
