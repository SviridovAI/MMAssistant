"""
Менеджер профилей аудиоустройств.

Предоставляет функциональность для создания, сохранения, загрузки и применения
профилей настроек аудиоустройств и микрофона.
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum


class UsageScenario(Enum):
    """Сценарии использования профиля."""

    DEFAULT = "default"
    STUDIO_RECORDING = "studio_recording"
    MEETING_CONFERENCE = "meeting_conference"
    STREAMING = "streaming"
    GAMING = "gaming"
    VOICE_OVER = "voice_over"
    CUSTOM = "custom"


@dataclass
class DeviceSettings:
    """Настройки аудиоустройств."""

    audio_output_device: str = ""
    audio_input_device: str = ""
    audio_output_device_id: int = -1
    audio_input_device_id: int = -1
    audio_auto_refresh_devices: bool = True


@dataclass
class MicrophoneSettings:
    """Настройки микрофона."""

    microphone_gain_db: float = 0.0
    microphone_max_gain: float = 100.0
    microphone_preview_enabled: bool = False
    microphone_preview_volume: float = 0.8
    audio_normalization_enabled: bool = False
    audio_normalization_peak_target: float = 0.5
    audio_normalization_level: float = -6.0


@dataclass
class ProfileMetadata:
    """Метаданные профиля."""

    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_used: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_modified: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    usage_count: int = 0
    device_fingerprint: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class AudioProfile:
    """Полный профиль аудионастроек."""

    name: str
    scenario: UsageScenario = UsageScenario.DEFAULT
    device_settings: DeviceSettings = field(default_factory=DeviceSettings)
    microphone_settings: MicrophoneSettings = field(default_factory=MicrophoneSettings)
    metadata: ProfileMetadata = field(default_factory=ProfileMetadata)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует профиль в словарь для сериализации."""
        return {
            "name": self.name,
            "scenario": self.scenario.value,
            "device_settings": asdict(self.device_settings),
            "microphone_settings": asdict(self.microphone_settings),
            "metadata": asdict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioProfile":
        """Создает профиль из словаря."""
        # Создаем вложенные объекты настроек
        device_settings = DeviceSettings(**data.get("device_settings", {}))
        microphone_settings = MicrophoneSettings(**data.get("microphone_settings", {}))

        # Обрабатываем метаданные
        metadata_data = data.get("metadata", {})
        metadata = ProfileMetadata(**metadata_data)

        # Получаем сценарий использования
        scenario_str = data.get("scenario", "default")
        try:
            scenario = UsageScenario(scenario_str)
        except ValueError:
            scenario = UsageScenario.DEFAULT

        return cls(
            name=data["name"],
            scenario=scenario,
            device_settings=device_settings,
            microphone_settings=microphone_settings,
            metadata=metadata,
        )

    def generate_fingerprint(self) -> str:
        """Генерирует уникальный отпечаток профиля на основе настроек устройств."""
        device_info = f"{self.device_settings.audio_output_device}_{self.device_settings.audio_input_device}"
        return hashlib.md5(device_info.encode()).hexdigest()[:8]

    def update_usage(self):
        """Обновляет метаданные при использовании профиля."""
        self.metadata.last_used = datetime.utcnow().isoformat() + "Z"
        self.metadata.usage_count += 1


class AudioProfileManager:
    """Менеджер профилей аудиоустройств."""

    def __init__(self, config_manager=None):
        """
        Инициализирует менеджер профилей.

        Args:
            config_manager: Экземпляр ConfigManager для доступа к конфигурации
        """
        self.config_manager = config_manager
        self.profiles: List[AudioProfile] = []
        self.current_profile: Optional[AudioProfile] = None
        self.profiles_file = "audio_profiles.json"
        self.default_profiles_created = False

        # Загружаем профили при инициализации
        self.load_profiles()

        # Создаем профили по умолчанию, если их нет
        if not self.profiles:
            self.create_default_profiles()

    def load_profiles(self) -> bool:
        """
        Загружает профили из файла.

        Returns:
            True если загрузка успешна, False в противном случае
        """
        try:
            if not os.path.exists(self.profiles_file):
                print(
                    f"Файл профилей {self.profiles_file} не найден, создаем пустой список"
                )
                self.profiles = []
                return True

            with open(self.profiles_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Проверяем формат файла
            if isinstance(data, dict) and "profiles" in data:
                profiles_data = data["profiles"]
                version = data.get("version", "1.0")
            else:
                # Старый формат (просто список профилей)
                profiles_data = data
                version = "1.0"

            self.profiles = []
            for profile_data in profiles_data:
                try:
                    profile = AudioProfile.from_dict(profile_data)
                    self.profiles.append(profile)
                except Exception as e:
                    print(
                        f"Ошибка загрузки профиля {profile_data.get('name', 'unknown')}: {e}"
                    )

            print(f"Загружено {len(self.profiles)} профилей (версия {version})")
            return True

        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON в файле {self.profiles_file}: {e}")
            self.profiles = []
            return False
        except Exception as e:
            print(f"Ошибка загрузки профилей: {e}")
            self.profiles = []
            return False

    def save_profiles(self) -> bool:
        """
        Сохраняет профили в файл.

        Returns:
            True если сохранение успешно, False в противном случае
        """
        try:
            # Обновляем отпечатки устройств для всех профилей
            for profile in self.profiles:
                if not profile.metadata.device_fingerprint:
                    profile.metadata.device_fingerprint = profile.generate_fingerprint()

            # Подготавливаем данные для сохранения
            data = {
                "version": "1.1",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "profiles": [profile.to_dict() for profile in self.profiles],
            }

            # Сохраняем с отступами для читаемости
            with open(self.profiles_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"Сохранено {len(self.profiles)} профилей в {self.profiles_file}")
            return True

        except Exception as e:
            print(f"Ошибка сохранения профилей: {e}")
            return False

    def create_default_profiles(self):
        """Создает набор профилей по умолчанию."""
        print("Создание профилей по умолчанию...")

        # Профиль по умолчанию
        default_profile = AudioProfile(
            name="По умолчанию",
            scenario=UsageScenario.DEFAULT,
            metadata=ProfileMetadata(
                description="Стандартные настройки системы", tags=["default", "system"]
            ),
        )

        # Профиль для студийной записи
        studio_profile = AudioProfile(
            name="Студийная запись",
            scenario=UsageScenario.STUDIO_RECORDING,
            microphone_settings=MicrophoneSettings(
                microphone_gain_db=0.0,
                microphone_max_gain=50.0,
                audio_normalization_enabled=True,
                audio_normalization_peak_target=0.7,
                audio_normalization_level=-3.0,
            ),
            metadata=ProfileMetadata(
                description="Высокое качество записи, низкое усиление, точный контроль уровней",
                tags=["studio", "recording", "high-quality"],
            ),
        )

        # Профиль для встреч/конференций
        meeting_profile = AudioProfile(
            name="Встречи/Конференции",
            scenario=UsageScenario.MEETING_CONFERENCE,
            microphone_settings=MicrophoneSettings(
                microphone_gain_db=10.0,
                microphone_max_gain=100.0,
                audio_normalization_enabled=True,
                audio_normalization_peak_target=0.5,
            ),
            metadata=ProfileMetadata(
                description="Оптимизировано для голосовой связи, среднее усиление",
                tags=["meeting", "conference", "voice"],
            ),
        )

        # Профиль для стриминга
        streaming_profile = AudioProfile(
            name="Стриминг",
            scenario=UsageScenario.STREAMING,
            microphone_settings=MicrophoneSettings(
                microphone_gain_db=15.0,
                microphone_max_gain=150.0,
                audio_normalization_enabled=True,
                audio_normalization_peak_target=0.6,
            ),
            metadata=ProfileMetadata(
                description="Оптимизировано для потоковой передачи, баланс качества и громкости",
                tags=["streaming", "broadcast", "live"],
            ),
        )

        # Профиль для игр
        gaming_profile = AudioProfile(
            name="Игры",
            scenario=UsageScenario.GAMING,
            microphone_settings=MicrophoneSettings(
                microphone_gain_db=20.0,
                microphone_max_gain=200.0,
                audio_normalization_enabled=False,
            ),
            metadata=ProfileMetadata(
                description="Высокое усиление для геймерских гарнитур",
                tags=["gaming", "headset", "high-gain"],
            ),
        )

        self.profiles = [
            default_profile,
            studio_profile,
            meeting_profile,
            streaming_profile,
            gaming_profile,
        ]

        self.default_profiles_created = True
        self.save_profiles()
        print(f"Создано {len(self.profiles)} профилей по умолчанию")

    def get_profile_names(self) -> List[str]:
        """Возвращает список имен всех профилей."""
        return [profile.name for profile in self.profiles]

    def get_profile(self, name: str) -> Optional[AudioProfile]:
        """Возвращает профиль по имени."""
        for profile in self.profiles:
            if profile.name == name:
                return profile
        return None

    def create_profile(
        self,
        name: str,
        scenario: UsageScenario = UsageScenario.CUSTOM,
        description: str = "",
        **settings,
    ) -> AudioProfile:
        """
        Создает новый профиль.

        Args:
            name: Название профиля
            scenario: Сценарий использования
            description: Описание профиля
            **settings: Дополнительные настройки для переопределения значений по умолчанию

        Returns:
            Созданный профиль
        """
        # Проверяем, существует ли уже профиль с таким именем
        if self.get_profile(name):
            raise ValueError(f"Профиль с именем '{name}' уже существует")

        # Создаем базовый профиль
        profile = AudioProfile(
            name=name,
            scenario=scenario,
            metadata=ProfileMetadata(description=description),
        )

        # Применяем переопределенные настройки
        if "device_settings" in settings:
            for key, value in settings["device_settings"].items():
                if hasattr(profile.device_settings, key):
                    setattr(profile.device_settings, key, value)

        if "microphone_settings" in settings:
            for key, value in settings["microphone_settings"].items():
                if hasattr(profile.microphone_settings, key):
                    setattr(profile.microphone_settings, key, value)

        # Добавляем профиль в список
        self.profiles.append(profile)

        # Сохраняем изменения
        self.save_profiles()

        print(f"Создан новый профиль: {name} ({scenario.value})")
        return profile

    def update_profile(self, name: str, **settings) -> bool:
        """
        Обновляет существующий профиль.

        Args:
            name: Имя профиля для обновления
            **settings: Настройки для обновления

        Returns:
            True если обновление успешно, False если профиль не найден
        """
        profile = self.get_profile(name)
        if not profile:
            return False

        # Обновляем основные поля
        if "scenario" in settings:
            try:
                profile.scenario = UsageScenario(settings["scenario"])
            except ValueError:
                pass

        if "description" in settings:
            profile.metadata.description = settings["description"]

        # Обновляем вложенные настройки
        for setting_type in [
            "device_settings",
            "microphone_settings",
        ]:
            if setting_type in settings:
                target_obj = getattr(profile, setting_type)
                for key, value in settings[setting_type].items():
                    if hasattr(target_obj, key):
                        setattr(target_obj, key, value)

        # Обновляем метаданные
        profile.metadata.last_modified = datetime.utcnow().isoformat() + "Z"

        # Сохраняем изменения
        self.save_profiles()

        print(f"Профиль обновлен: {name}")
        return True

    def delete_profile(self, name: str) -> bool:
        """
        Удаляет профиль.

        Args:
            name: Имя профиля для удаления

        Returns:
            True если удаление успешно, False если профиль не найден
        """
        profile = self.get_profile(name)
        if not profile:
            return False

        # Не позволяем удалить последний профиль
        if len(self.profiles) <= 1:
            print("Нельзя удалить последний профиль")
            return False

        # Удаляем профиль из списка
        self.profiles = [p for p in self.profiles if p.name != name]

        # Если удаляемый профиль был текущим, выбираем другой
        if self.current_profile and self.current_profile.name == name:
            self.current_profile = self.profiles[0] if self.profiles else None

        # Сохраняем изменения
        self.save_profiles()

        print(f"Профиль удален: {name}")
        return True

    def apply_profile(self, name: str, config: Optional[Dict] = None) -> bool:
        """
        Применяет настройки профиля к конфигурации.

        Args:
            name: Имя профиля для применения
            config: Словарь конфигурации для обновления (опционально)

        Returns:
            True если применение успешно, False если профиль не найден
        """
        profile = self.get_profile(name)
        if not profile:
            return False

        # Обновляем метаданные использования
        profile.update_usage()
        self.current_profile = profile

        # Если передан словарь конфигурации, обновляем его
        if config:
            # Обновляем настройки устройств
            for key, value in asdict(profile.device_settings).items():
                config[key] = value

            # Обновляем настройки микрофона
            for key, value in asdict(profile.microphone_settings).items():
                config[key] = value

            # Сохраняем информацию о текущем профиле
            config["current_audio_profile"] = profile.name
            config["audio_profile_scenario"] = profile.scenario.value

        # Сохраняем изменения в профилях
        self.save_profiles()

        print(f"Применен профиль: {name} ({profile.scenario.value})")
        return True

    def export_profile(self, name: str, filepath: str) -> bool:
        """
        Экспортирует профиль в отдельный файл.

        Args:
            name: Имя профиля для экспорта
            filepath: Путь к файлу для сохранения

        Returns:
            True если экспорт успешен, False если профиль не найден
        """
        profile = self.get_profile(name)
        if not profile:
            return False

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)

            print(f"Профиль {name} экспортирован в {filepath}")
            return True

        except Exception as e:
            print(f"Ошибка экспорта профиля: {e}")
            return False

    def import_profile(self, filepath: str) -> Optional[AudioProfile]:
        """
        Импортирует профиль из файла.

        Args:
            filepath: Путь к файлу профиля

        Returns:
            Импортированный профиль или None при ошибке
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Создаем профиль из данных
            profile = AudioProfile.from_dict(data)

            # Проверяем, существует ли уже профиль с таким именем
            existing = self.get_profile(profile.name)
            if existing:
                # Добавляем суффикс для уникальности
                base_name = profile.name
                counter = 1
                while self.get_profile(f"{base_name}_{counter}"):
                    counter += 1
                profile.name = f"{base_name}_{counter}"

            # Добавляем профиль в список
            self.profiles.append(profile)

            # Сохраняем изменения
            self.save_profiles()

            print(f"Профиль импортирован: {profile.name}")
            return profile

        except Exception as e:
            print(f"Ошибка импорта профиля: {e}")
            return None

    def get_profile_by_scenario(self, scenario: UsageScenario) -> List[AudioProfile]:
        """
        Возвращает профили для указанного сценария использования.

        Args:
            scenario: Сценарий использования

        Returns:
            Список профилей для этого сценария
        """
        return [p for p in self.profiles if p.scenario == scenario]

    def get_current_profile(self) -> Optional[AudioProfile]:
        """Возвращает текущий активный профиль."""
        return self.current_profile

    def set_current_profile(self, name: str) -> bool:
        """
        Устанавливает текущий профиль без применения его настроек.

        Args:
            name: Имя профиля

        Returns:
            True если успешно, False если профиль не найден
        """
        profile = self.get_profile(name)
        if not profile:
            return False

        self.current_profile = profile
        return True

    def create_profile_from_current_config(
        self, name: str, config: Dict
    ) -> AudioProfile:
        """
        Создает профиль из текущей конфигурации.

        Args:
            name: Имя нового профиля
            config: Текущая конфигурация

        Returns:
            Созданный профиль
        """
        # Создаем объекты настроек из конфигурации
        device_settings = DeviceSettings(
            audio_output_device=config.get("audio_output_device", ""),
            audio_input_device=config.get("audio_input_device", ""),
            audio_output_device_id=config.get("audio_output_device_id", -1),
            audio_input_device_id=config.get("audio_input_device_id", -1),
            audio_auto_refresh_devices=config.get("audio_auto_refresh_devices", True),
        )

        microphone_settings = MicrophoneSettings(
            microphone_gain_db=config.get("microphone_gain_db", 0.0),
            microphone_max_gain=config.get("microphone_max_gain", 100.0),
            microphone_preview_enabled=config.get("microphone_preview_enabled", False),
            microphone_preview_volume=config.get("microphone_preview_volume", 0.8),
            audio_normalization_enabled=config.get(
                "audio_normalization_enabled", False
            ),
            audio_normalization_peak_target=config.get(
                "audio_normalization_peak_target", 0.5
            ),
            audio_normalization_level=config.get("audio_normalization_level", -6.0),
        )

        # Создаем профиль
        profile = AudioProfile(
            name=name,
            scenario=UsageScenario.CUSTOM,
            device_settings=device_settings,
            microphone_settings=microphone_settings,
            metadata=ProfileMetadata(
                description="Создан из текущей конфигурации",
                tags=["custom", "from-config"],
            ),
        )

        # Добавляем профиль в список
        self.profiles.append(profile)

        # Сохраняем изменения
        self.save_profiles()

        print(f"Создан профиль из текущей конфигурации: {name}")
        return profile
