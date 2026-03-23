import os
import json
import keyring
import keyring.errors
from pathlib import Path

# Константы
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    # LLM (OpenAI-совместимые API)
    "llm_api_url": "https://api.deepseek.com/v1/chat/completions",
    "llm_model": "deepseek-chat",
    "llm_temperature": 0.7,
    # Whisper
    "whisper_url": "http://192.168.1.95:9000/asr",
    "whisper_language": "ru",
    "whisper_word_timestamps": False,
    "whisper_remove_words": True,
    # Папки
    "recordings_folder": "",
    "whisper_output_folder": "",
    "default_audio_folder": str(Path.home()),
    "default_prompt_folder": str(Path.home()),
    "default_output_folder": str(Path.home() / "WhisperDeepseekOutputs"),
    # Аудио нормализация
    "audio_normalization_enabled": True,
    "audio_normalization_level": -6.0,  # dBFS
    "audio_normalization_peak_target": 0.5,  # пиковое значение (0.0-1.0)
}


class ConfigManager:
    """Менеджер конфигурации приложения."""

    def __init__(self, config_file=None):
        self.config_file = config_file or CONFIG_FILE
        self.config = None
        self.load()

    def load(self):
        """Загружает конфигурацию из файла или создает новую."""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = DEFAULT_CONFIG.copy()
            self.save()
        return self.config

    def save(self, config=None):
        """Сохраняет конфигурацию в файл."""
        if config is not None:
            self.config = config
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)

    def get(self, key, default=None):
        """Получает значение из конфигурации."""
        return self.config.get(key, default)

    def set(self, key, value):
        """Устанавливает значение в конфигурации."""
        self.config[key] = value

    def update(self, updates):
        """Обновляет несколько значений в конфигурации."""
        self.config.update(updates)

    def get_api_key(self):
        """Получает API ключ из keyring. Возвращает None если ключа нет."""
        try:
            key = keyring.get_password("MMAssistant", "llm_api_key")
            return key if key else None
        except keyring.errors.KeyringError as e:
            print(f"Ошибка keyring: {e}")
            raise RuntimeError(f"Не удалось получить API ключ из keyring: {e}")

    def set_api_key(self, api_key):
        """Сохраняет API ключ в keyring."""
        if not api_key:
            raise ValueError("API ключ не может быть пустым")
        try:
            keyring.set_password("MMAssistant", "llm_api_key", api_key)
            return True
        except keyring.errors.KeyringError as e:
            print(f"Ошибка сохранения в keyring: {e}")
            raise RuntimeError(f"Не удалось сохранить API ключ в keyring: {e}")

    def has_api_key(self):
        """Проверяет, есть ли сохраненный API ключ."""
        try:
            key = keyring.get_password("MMAssistant", "llm_api_key")
            return bool(key)
        except:
            return False

    def migrate_old_config(self):
        """
        Мигрирует старую конфигурацию: удаляет ключ из config.json если он там есть.
        Ключ должен быть введен заново через настройки.
        """
        if os.path.exists(self.config_file):
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Если есть ключ в config.json, удаляем его
            if "llm_api_key" in config:
                print("Удаление старого API ключа из config.json...")
                del config["llm_api_key"]
                self.save(config)
                return True
        return False


# Функции-обертки для обратной совместимости с существующим кодом
_config_manager = ConfigManager()


def load_config():
    """Загружает конфигурацию (обратная совместимость)."""
    return _config_manager.load()


def save_config(config):
    """Сохраняет конфигурацию (обратная совместимость)."""
    _config_manager.save(config)


def get_api_key():
    """Получает API ключ (обратная совместимость)."""
    return _config_manager.get_api_key()


def set_api_key(api_key):
    """Сохраняет API ключ (обратная совместимость)."""
    return _config_manager.set_api_key(api_key)


def has_api_key():
    """Проверяет наличие API ключа (обратная совместимость)."""
    return _config_manager.has_api_key()


def migrate_old_config():
    """Мигрирует старую конфигурацию (обратная совместимость)."""
    return _config_manager.migrate_old_config()
