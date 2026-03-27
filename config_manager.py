import os
import json
import time
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
    "whisper_url": "http://127.0.0.1:9000/asr",
    "whisper_language": "ru",
    "whisper_word_timestamps": False,
    "whisper_remove_words": True,
    # Папки
    "recordings_folder": "",
    "whisper_output_folder": "",
    "default_audio_folder": str(Path.home()),
    "default_prompt_folder": str(Path.home()),
    "default_output_folder": str(Path.home() / "MMAssistantResults"),
    # Аудио нормализация
    "audio_normalization_enabled": False,
    "audio_normalization_level": -6.0,  # dBFS
    "audio_normalization_peak_target": 0.5,  # пиковое значение (0.0-1.0),
    # Аудио устройства
    "audio_output_device": "",  # имя выходного устройства
    "audio_input_device": "",  # имя входного устройства
    "audio_output_device_id": -1,  # индекс выходного устройства
    "audio_input_device_id": -1,  # индекс входного устройства
    "audio_auto_refresh_devices": True,  # автообновление списка устройств
    "audio_last_refresh": 0.0,  # время последнего обновления (timestamp)
    "audio_cache_ttl": 30.0,  # время жизни кэша устройств в секундах
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
            raise RuntimeError(f"Не удалось получить API ключ из keyring: {e}")

    def set_api_key(self, api_key):
        """Сохраняет API ключ в keyring."""
        if not api_key:
            raise ValueError("API ключ не может быть пустым")
        try:
            keyring.set_password("MMAssistant", "llm_api_key", api_key)
            return True
        except keyring.errors.KeyringError as e:
            raise RuntimeError(f"Не удалось сохранить API ключ в keyring: {e}")

    def has_api_key(self):
        """Проверяет, есть ли сохраненный API ключ."""
        try:
            key = keyring.get_password("MMAssistant", "llm_api_key")
            return bool(key)
        except:
            return False

    # Методы для работы с аудио настройками
    def get_audio_output_device(self):
        """Возвращает настроенное выходное аудиоустройство."""
        return self.get("audio_output_device", "")

    def get_audio_input_device(self):
        """Возвращает настроенное входное аудиоустройство."""
        return self.get("audio_input_device", "")

    def get_audio_output_device_id(self):
        """Возвращает индекс настроенного выходного аудиоустройства."""
        return self.get("audio_output_device_id", -1)

    def get_audio_input_device_id(self):
        """Возвращает индекс настроенного входного аудиоустройства."""
        return self.get("audio_input_device_id", -1)

    def set_audio_output_device(self, device_name, device_id=-1):
        """Устанавливает выходное аудиоустройство."""
        self.set("audio_output_device", device_name)
        if device_id >= 0:
            self.set("audio_output_device_id", device_id)
        self.save()

    def set_audio_input_device(self, device_name, device_id=-1):
        """Устанавливает входное аудиоустройство."""
        self.set("audio_input_device", device_name)
        if device_id >= 0:
            self.set("audio_input_device_id", device_id)
        self.save()

    def get_audio_auto_refresh(self):
        """Возвращает настройку автообновления списка устройств."""
        return self.get("audio_auto_refresh_devices", True)

    def set_audio_auto_refresh(self, enabled):
        """Устанавливает настройку автообновления списка устройств."""
        self.set("audio_auto_refresh_devices", bool(enabled))
        self.save()

    def update_audio_last_refresh(self):
        """Обновляет время последнего обновления списка устройств."""
        import time

        self.set("audio_last_refresh", time.time())
        self.save()

    def get_audio_last_refresh(self):
        """Возвращает время последнего обновления списка устройств."""
        return self.get("audio_last_refresh", 0.0)

    def should_refresh_devices(self):
        """Определяет, нужно ли обновлять список устройств на основе TTL."""
        import time

        last_refresh = self.get_audio_last_refresh()
        cache_ttl = self.get("audio_cache_ttl", 30.0)
        auto_refresh = self.get_audio_auto_refresh()

        if not auto_refresh:
            return False

        if last_refresh == 0:
            return True

        return (time.time() - last_refresh) > cache_ttl

    def validate_audio_device(self, device_type, device_name=None, device_id=None):
        """
        Валидирует аудиоустройство.

        Args:
            device_type: "input" или "output"
            device_name: имя устройства для проверки
            device_id: индекс устройства для проверки

        Returns:
            tuple: (is_valid, error_message)
        """
        # Импортируем здесь, чтобы избежать циклических зависимостей
        from audio_recorder import AudioRecorder

        recorder = AudioRecorder(lambda *args: None, self.config)
        try:
            devices_info = recorder.get_available_devices()

            if device_type == "input":
                devices = devices_info.get("input_devices", [])
                default_idx = devices_info.get("default_input_index")
            else:
                devices = devices_info.get("output_devices", [])
                default_idx = devices_info.get("default_output_index")

            # Если не указано устройство, используем устройство по умолчанию
            if not device_name and device_id is None:
                if default_idx is not None:
                    return (
                        True,
                        f"Используется устройство по умолчанию (индекс {default_idx})",
                    )
                else:
                    return False, "Устройство по умолчанию не найдено"

            # Поиск по индексу
            if device_id is not None:
                for device in devices:
                    if device.get("index") == device_id:
                        return True, f"Устройство найдено: {device.get('name')}"
                return False, f"Устройство с индексом {device_id} не найдено"

            # Поиск по имени
            if device_name:
                for device in devices:
                    if device.get("name") == device_name:
                        return True, f"Устройство найдено: {device_name}"
                return False, f"Устройство с именем '{device_name}' не найдено"

            return False, "Не указаны параметры устройства"

        except Exception as e:
            return False, f"Ошибка при валидации устройства: {e}"

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


# Вспомогательные функции для работы с аудиоустройствами
def get_default_audio_device(device_type="output"):
    """
    Возвращает устройство по умолчанию для указанного типа.

    Args:
        device_type: "input" или "output"

    Returns:
        dict: Информация об устройстве по умолчанию или None
    """
    from audio_recorder import AudioRecorder

    recorder = AudioRecorder(lambda *args: None, _config_manager.config)
    try:
        devices_info = recorder.get_available_devices()

        if device_type == "input":
            default_idx = devices_info.get("default_input_index")
            devices = devices_info.get("input_devices", [])
        else:
            default_idx = devices_info.get("default_output_index")
            devices = devices_info.get("output_devices", [])

        if default_idx is not None:
            for device in devices:
                if device.get("index") == default_idx:
                    return device

        # Если устройство по умолчанию не найдено, возвращаем первое доступное
        if devices:
            return devices[0]

        return None
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(
            f"Ошибка при получении устройства по умолчанию: {e}"
        )
        return None


def validate_audio_device_selection(
    device_name=None, device_id=None, device_type="output"
):
    """
    Валидирует выбранное аудиоустройство.

    Args:
        device_name: имя устройства
        device_id: индекс устройства
        device_type: "input" или "output"

    Returns:
        tuple: (is_valid, device_info, error_message)
    """
    from audio_recorder import AudioRecorder

    recorder = AudioRecorder(lambda *args: None, _config_manager.config)
    try:
        devices_info = recorder.get_available_devices()

        if device_type == "input":
            devices = devices_info.get("input_devices", [])
        else:
            devices = devices_info.get("output_devices", [])

        # Если не указано устройство, используем устройство по умолчанию
        if not device_name and device_id is None:
            default_device = get_default_audio_device(device_type)
            if default_device:
                return True, default_device, "Используется устройство по умолчанию"
            else:
                return False, None, "Устройство по умолчанию не найдено"

        # Поиск по индексу
        if device_id is not None:
            for device in devices:
                if device.get("index") == device_id:
                    return True, device, f"Устройство найдено по индексу {device_id}"
            return False, None, f"Устройство с индексом {device_id} не найдено"

        # Поиск по имени
        if device_name:
            for device in devices:
                if device.get("name") == device_name:
                    return True, device, f"Устройство найдено по имени '{device_name}'"
            return False, None, f"Устройство с именем '{device_name}' не найдено"

        return False, None, "Не указаны параметры устройства"

    except Exception as e:
        return False, None, f"Ошибка при валидации устройства: {e}"


def refresh_audio_devices_list(force=False):
    """
    Обновляет список аудиоустройств.

    Args:
        force: если True, игнорирует кэш и автообновление

    Returns:
        dict: Обновленная информация об устройствах или None при ошибке
    """
    from audio_recorder import AudioRecorder

    # Проверяем, нужно ли обновлять
    if not force and not _config_manager.should_refresh_devices():
        return None

    recorder = AudioRecorder(lambda *args: None, _config_manager.config)
    try:
        devices_info = recorder.get_available_devices(force_refresh=True)
        _config_manager.update_audio_last_refresh()
        return devices_info
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(
            f"Ошибка при обновлении списка устройств: {e}"
        )
        return None


def get_audio_devices_summary():
    """
    Возвращает краткую сводку по доступным аудиоустройствам.

    Returns:
        dict: Сводка с количеством устройств и статусом
    """
    from audio_recorder import AudioRecorder

    recorder = AudioRecorder(lambda *args: None, _config_manager.config)
    try:
        devices_info = recorder.get_available_devices()

        summary = {
            "output_count": len(devices_info.get("output_devices", [])),
            "input_count": len(devices_info.get("input_devices", [])),
            "has_default_output": devices_info.get("default_output_index") is not None,
            "has_default_input": devices_info.get("default_input_index") is not None,
            "cache_age": time.time() - devices_info.get("cache_timestamp", 0),
            "errors": len(devices_info.get("errors", [])),
            "timestamp": devices_info.get("cache_timestamp", 0),
        }

        return summary
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(f"Ошибка при получении сводки устройств: {e}")
        return {
            "output_count": 0,
            "input_count": 0,
            "has_default_output": False,
            "has_default_input": False,
            "cache_age": 0,
            "errors": 1,
            "timestamp": 0,
        }
