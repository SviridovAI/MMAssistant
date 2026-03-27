import os
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import requests

# Импорт модуля конфигурации
from config_manager import (
    load_config,
    save_config,
    get_api_key,
    set_api_key,
    has_api_key,
    migrate_old_config,
    DEFAULT_CONFIG,
)


def get_available_models(api_url, api_key):
    """
    Запрашивает список доступных моделей из OpenAI-совместимого API.

    Args:
        api_url: Базовый URL API (например, https://api.deepseek.com/v1)
        api_key: API ключ

    Returns:
        List[str]: Список идентификаторов моделей
    """
    # Нормализуем URL (убираем конечный /chat/completions если есть)
    base_url = api_url.rstrip("/")
    if "/chat/completions" in base_url:
        base_url = base_url.split("/chat/completions")[0]

    models_url = f"{base_url}/models"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.get(models_url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Разные форматы ответа:
        # OpenAI: {"data": [{"id": "gpt-4", ...}, ...]}
        # DeepSeek: {"data": [{"id": "deepseek-chat", ...}, ...]}
        if "data" in data and isinstance(data["data"], list):
            models = [model["id"] for model in data["data"]]
            return sorted(models)
        else:
            # Альтернативный формат
            return []

    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка сети: {e}")
    except json.JSONDecodeError:
        raise Exception("Неверный формат ответа от API")
    except Exception as e:
        raise Exception(f"Ошибка: {e}")


class LLMTab:
    """Вкладка настроек LLM."""

    def __init__(self, parent, config, api_key):
        self.parent = parent
        self.config = config
        self.api_key = api_key
        self.models = []  # Список доступных моделей

        self.create_widgets()
        self.load_current_values()

    def create_widgets(self):
        """Создает все элементы управления вкладки."""
        # URL endpoint
        ttk.Label(self.parent, text="URL endpoint модели:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )

        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(self.parent, textvariable=self.url_var, width=50)
        self.url_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        # Ключ доступа
        ttk.Label(self.parent, text="Ключ доступа:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )

        self.key_var = tk.StringVar()
        self.key_entry = ttk.Entry(
            self.parent, textvariable=self.key_var, show="*", width=40
        )
        self.key_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # Кнопка показать/скрыть
        self.show_key_var = tk.BooleanVar(value=False)
        self.show_key_btn = ttk.Checkbutton(
            self.parent,
            text="Показать",
            variable=self.show_key_var,
            command=self.toggle_key_visibility,
        )
        self.show_key_btn.grid(row=1, column=2, padx=5, pady=5)

        # Модель LLM
        ttk.Label(self.parent, text="Модель LLM:").grid(
            row=2, column=0, sticky="w", padx=5, pady=5
        )

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            self.parent, textvariable=self.model_var, state="readonly", width=30
        )
        self.model_combo.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        # Кнопка обновить список
        self.refresh_btn = ttk.Button(
            self.parent, text="Обновить список", command=self.refresh_models
        )
        self.refresh_btn.grid(row=2, column=2, padx=5, pady=5)

        # Температура
        ttk.Label(self.parent, text="Температура (0.0-1.0):").grid(
            row=3, column=0, sticky="w", padx=5, pady=5
        )

        self.temp_var = tk.DoubleVar()
        self.temp_scale = ttk.Scale(
            self.parent,
            from_=0.0,
            to=1.0,
            variable=self.temp_var,
            orient="horizontal",
            command=self.on_temp_change,
        )
        self.temp_scale.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

        self.temp_label = ttk.Label(self.parent, text="0.7")
        self.temp_label.grid(row=3, column=2, padx=5, pady=5)

        # Статус
        self.status_var = tk.StringVar(value="Готов")
        self.status_label = ttk.Label(self.parent, textvariable=self.status_var)
        self.status_label.grid(
            row=4, column=0, columnspan=3, sticky="w", padx=5, pady=10
        )

        # Настройка весов колонок
        self.parent.grid_columnconfigure(1, weight=1)

    def load_current_values(self):
        """Загружает текущие значения из конфигурации."""
        # URL
        self.url_var.set(
            self.config.get(
                "llm_api_url", "https://api.deepseek.com/v1/chat/completions"
            )
        )

        # Ключ (показываем звездочки если есть ключ)
        if self.api_key:
            self.key_var.set("*" * 20)
        else:
            self.key_var.set("")

        # Модель
        current_model = self.config.get("llm_model", "deepseek-chat")
        self.model_var.set(current_model)

        # Загружаем список моделей (может быть пустым)
        self.models = [current_model]
        self.model_combo["values"] = self.models

        # Температура
        temperature = self.config.get("llm_temperature", 0.7)
        self.temp_var.set(temperature)
        self.temp_label.config(text=f"{temperature:.1f}")

    def toggle_key_visibility(self):
        """Переключает видимость API ключа."""
        if self.show_key_var.get():
            self.key_entry.config(show="")
            if self.api_key and self.key_var.get() == "*" * 20:
                self.key_var.set(self.api_key)
        else:
            self.key_entry.config(show="*")
            if self.api_key and self.key_var.get() == self.api_key:
                self.key_var.set("*" * 20)

    def on_temp_change(self, value):
        """Обработчик изменения температуры."""
        temp = float(value)
        self.temp_label.config(text=f"{temp:.1f}")

    def refresh_models(self):
        """Обновляет список доступных моделей из API."""
        url = self.url_var.get().strip()
        key = self.get_current_key()

        if not url:
            self.set_status("Ошибка: URL не указан", "error")
            return

        if not key:
            self.set_status("Ошибка: ключ не указан", "error")
            return

        self.set_status("Запрос списка моделей...", "info")
        self.refresh_btn.config(state="disabled")

        # Запускаем в отдельном потоке чтобы не блокировать GUI
        thread = threading.Thread(target=self._fetch_models, args=(url, key))
        thread.daemon = True
        thread.start()

    def _fetch_models(self, url, key):
        """Фоновая задача для получения списка моделей."""
        try:
            models = get_available_models(url, key)

            # Обновляем GUI в основном потоке
            self.parent.after(0, self._update_models_list, models)
            self.parent.after(
                0,
                lambda: self.set_status(f"Загружено {len(models)} моделей", "success"),
            )

        except Exception as e:
            self.parent.after(0, lambda: self.set_status(f"Ошибка: {str(e)}", "error"))
        finally:
            self.parent.after(0, lambda: self.refresh_btn.config(state="normal"))

    def _update_models_list(self, models):
        """Обновляет список моделей в Combobox."""
        if models:
            self.models = models
            self.model_combo["values"] = models

            # Если текущая модель не в списке, добавляем её
            current_model = self.model_var.get()
            if current_model not in models:
                self.models.insert(0, current_model)
                self.model_combo["values"] = self.models
        else:
            self.set_status("Не удалось получить список моделей", "warning")

    def get_current_key(self):
        """Возвращает текущий введенный ключ."""
        key = self.key_var.get()
        # Если показываются звездочки, используем сохраненный ключ
        if key == "*" * 20 and self.api_key:
            return self.api_key
        return key

    def set_status(self, message, status_type="info"):
        """Устанавливает статусное сообщение."""
        self.status_var.set(message)

        colors = {
            "info": "black",
            "success": "green",
            "warning": "orange",
            "error": "red",
        }

        color = colors.get(status_type, "black")
        self.status_label.config(foreground=color)

    def get_values(self):
        """Возвращает текущие значения настроек."""
        return {
            "llm_api_url": self.url_var.get().strip(),
            "llm_model": self.model_var.get(),
            "llm_temperature": round(self.temp_var.get(), 1),
        }

    def get_api_key(self):
        """Возвращает введенный API ключ (или None если не менялся)."""
        current_key = self.key_var.get()

        # Если показываются звездочки и у нас есть сохраненный ключ
        if current_key == "*" * 20 and self.api_key:
            return None  # Ключ не менялся

        # Если поле пустое
        if not current_key:
            return ""

        # Если показываются звездочки но нет сохраненного ключа
        if current_key == "*" * 20 and not self.api_key:
            return ""

        # Новый ключ
        return current_key


class WhisperTab:
    """Вкладка настроек Whisper."""

    def __init__(self, parent, config):
        self.parent = parent
        self.config = config

        self.create_widgets()
        self.load_current_values()

    def create_widgets(self):
        """Создает все элементы управления вкладки."""
        row = 0

        # URL endpoint Whisper
        ttk.Label(self.parent, text="URL endpoint Whisper:").grid(
            row=row, column=0, sticky="w", padx=5, pady=5
        )

        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(self.parent, textvariable=self.url_var, width=50)
        self.url_entry.grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=5
        )
        row += 1

        # Временные метки для слов
        self.word_timestamps_var = tk.BooleanVar()
        self.word_timestamps_cb = ttk.Checkbutton(
            self.parent,
            text="Включить временные метки для слов",
            variable=self.word_timestamps_var,
        )
        self.word_timestamps_cb.grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=5
        )
        row += 1

        # Подсказка для временных меток
        hint_label = ttk.Label(
            self.parent,
            text="(добавляет информацию о времени начала/конца каждого слова в результат)",
            font=("TkDefaultFont", 8),
        )
        hint_label.grid(
            row=row, column=0, columnspan=3, sticky="w", padx=20, pady=(0, 10)
        )
        row += 1

        # Язык распознавания
        ttk.Label(self.parent, text="Язык распознавания:").grid(
            row=row, column=0, sticky="w", padx=5, pady=5
        )

        self.language_var = tk.StringVar()
        self.language_combo = ttk.Combobox(
            self.parent,
            textvariable=self.language_var,
            values=["ru", "en", "auto", "de", "fr", "es", "it", "zh"],
            state="readonly",
            width=10,
        )
        self.language_combo.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1

        # Удаление слов из результата
        self.remove_words_var = tk.BooleanVar()
        self.remove_words_cb = ttk.Checkbutton(
            self.parent,
            text="Удалять слова из результата",
            variable=self.remove_words_var,
        )
        self.remove_words_cb.grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=5
        )
        row += 1

        # Подсказка для удаления слов
        remove_hint = ttk.Label(
            self.parent,
            text="(оставляет только текст, уменьшает размер JSON файлов)",
            font=("TkDefaultFont", 8),
        )
        remove_hint.grid(
            row=row, column=0, columnspan=3, sticky="w", padx=20, pady=(0, 10)
        )
        row += 1

        # Кнопка тестирования соединения
        self.test_btn = ttk.Button(
            self.parent, text="Проверить соединение", command=self.test_connection
        )
        self.test_btn.grid(row=row, column=0, padx=5, pady=10)

        # Статус тестирования
        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(self.parent, textvariable=self.status_var)
        self.status_label.grid(
            row=row, column=1, columnspan=2, sticky="w", padx=5, pady=10
        )

        # Настройка весов колонок
        self.parent.grid_columnconfigure(1, weight=1)

    def load_current_values(self):
        """Загружает текущие значения из конфигурации."""
        # URL
        self.url_var.set(self.config.get("whisper_url", "http://192.168.1.95:9000/asr"))

        # Временные метки
        self.word_timestamps_var.set(self.config.get("whisper_word_timestamps", False))

        # Язык
        self.language_var.set(self.config.get("whisper_language", "ru"))

        # Удаление слов
        self.remove_words_var.set(self.config.get("whisper_remove_words", True))

    def test_connection(self):
        """Проверяет соединение с Whisper endpoint."""
        url = self.url_var.get().strip()

        if not url:
            self.set_status("Ошибка: URL не указан", "error")
            return

        self.set_status("Проверка соединения...", "info")
        self.test_btn.config(state="disabled")

        # Запускаем в отдельном потоке
        thread = threading.Thread(target=self._test_connection_thread, args=(url,))
        thread.daemon = True
        thread.start()

    def _test_connection_thread(self, url):
        """Фоновая задача для проверки соединения."""
        try:
            # Простой GET запрос к endpoint
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                self.parent.after(
                    0, lambda: self.set_status("✅ Соединение установлено", "success")
                )
            else:
                self.parent.after(
                    0,
                    lambda: self.set_status(
                        f"⚠️ Endpoint ответил с кодом {response.status_code}", "warning"
                    ),
                )

        except requests.exceptions.ConnectionError:
            self.parent.after(
                0,
                lambda: self.set_status(
                    "❌ Не удалось подключиться к endpoint", "error"
                ),
            )
        except requests.exceptions.Timeout:
            self.parent.after(
                0, lambda: self.set_status("❌ Таймаут соединения", "error")
            )
        except Exception as e:
            self.parent.after(
                0, lambda: self.set_status(f"❌ Ошибка: {str(e)}", "error")
            )
        finally:
            self.parent.after(0, lambda: self.test_btn.config(state="normal"))

    def set_status(self, message, status_type="info"):
        """Устанавливает статусное сообщение."""
        self.status_var.set(message)

        colors = {
            "info": "black",
            "success": "green",
            "warning": "orange",
            "error": "red",
        }

        color = colors.get(status_type, "black")
        self.status_label.config(foreground=color)

    def get_values(self):
        """Возвращает текущие значения настроек."""
        return {
            "whisper_url": self.url_var.get().strip(),
            "whisper_word_timestamps": self.word_timestamps_var.get(),
            "whisper_language": self.language_var.get(),
            "whisper_remove_words": self.remove_words_var.get(),
        }


class FoldersTab:
    """Вкладка настроек папок."""

    def __init__(self, parent, config):
        self.parent = parent
        self.config = config

        self.create_widgets()
        self.load_current_values()

    def create_widgets(self):
        """Создает все элементы управления вкладки."""
        row = 0

        # 1. Папка для записи аудио
        ttk.Label(self.parent, text="Папка для записи аудио:").grid(
            row=row, column=0, sticky="w", padx=5, pady=5
        )

        self.recordings_var = tk.StringVar()
        self.recordings_entry = ttk.Entry(
            self.parent, textvariable=self.recordings_var, width=50
        )
        self.recordings_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)

        self.recordings_btn = ttk.Button(
            self.parent,
            text="Обзор...",
            command=lambda: self.browse_folder(self.recordings_var),
        )
        self.recordings_btn.grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # 2. Папка промтов
        ttk.Label(self.parent, text="Папка промтов:").grid(
            row=row, column=0, sticky="w", padx=5, pady=5
        )

        self.prompt_var = tk.StringVar()
        self.prompt_entry = ttk.Entry(
            self.parent, textvariable=self.prompt_var, width=50
        )
        self.prompt_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)

        self.prompt_btn = ttk.Button(
            self.parent,
            text="Обзор...",
            command=lambda: self.browse_folder(self.prompt_var),
        )
        self.prompt_btn.grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # 3. Папка контекста
        ttk.Label(self.parent, text="Папка контекста:").grid(
            row=row, column=0, sticky="w", padx=5, pady=5
        )

        self.context_var = tk.StringVar()
        self.context_entry = ttk.Entry(
            self.parent, textvariable=self.context_var, width=50
        )
        self.context_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)

        self.context_btn = ttk.Button(
            self.parent,
            text="Обзор...",
            command=lambda: self.browse_folder(self.context_var),
        )
        self.context_btn.grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # 4. Папка результатов
        ttk.Label(self.parent, text="Папка результатов:").grid(
            row=row, column=0, sticky="w", padx=5, pady=5
        )

        self.output_var = tk.StringVar()
        self.output_entry = ttk.Entry(
            self.parent, textvariable=self.output_var, width=50
        )
        self.output_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)

        self.output_btn = ttk.Button(
            self.parent,
            text="Обзор...",
            command=lambda: self.browse_folder(self.output_var),
        )
        self.output_btn.grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # 5. Папка для вывода Whisper (опционально)
        ttk.Label(self.parent, text="Папка для JSON Whisper (опционально):").grid(
            row=row, column=0, sticky="w", padx=5, pady=5
        )

        self.whisper_output_var = tk.StringVar()
        self.whisper_output_entry = ttk.Entry(
            self.parent, textvariable=self.whisper_output_var, width=50
        )
        self.whisper_output_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)

        self.whisper_output_btn = ttk.Button(
            self.parent,
            text="Обзор...",
            command=lambda: self.browse_folder(self.whisper_output_var),
        )
        self.whisper_output_btn.grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # Кнопка создания папок
        self.create_folders_btn = ttk.Button(
            self.parent, text="Создать все папки", command=self.create_all_folders
        )
        self.create_folders_btn.grid(row=row, column=0, columnspan=3, pady=20)

        # Статус
        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(self.parent, textvariable=self.status_var)
        self.status_label.grid(
            row=row + 1, column=0, columnspan=3, sticky="w", padx=5, pady=5
        )

        # Настройка весов колонок
        self.parent.grid_columnconfigure(1, weight=1)

    def load_current_values(self):
        """Загружает текущие значения из конфигурации."""
        # Папка для записи аудио
        self.recordings_var.set(
            self.config.get("recordings_folder", "")
            or self.config.get("default_audio_folder", str(Path.home()))
        )

        # Папка промтов
        self.prompt_var.set(self.config.get("default_prompt_folder", str(Path.home())))

        # Папка контекста (используем ту же что и для промтов)
        self.context_var.set(self.config.get("default_prompt_folder", str(Path.home())))

        # Папка результатов
        self.output_var.set(
            self.config.get(
                "default_output_folder", str(Path.home() / "WhisperDeepseekOutputs")
            )
        )

        # Папка для JSON Whisper
        self.whisper_output_var.set(self.config.get("whisper_output_folder", ""))

    def browse_folder(self, var):
        """Открывает диалог выбора папки и обновляет переменную."""
        initial_dir = var.get() or str(Path.home())

        folder = filedialog.askdirectory(title="Выберите папку", initialdir=initial_dir)

        if folder:
            var.set(folder)

    def create_all_folders(self):
        """Создает все указанные папки если они не существуют."""
        folders = [
            ("Папка для записи аудио", self.recordings_var.get()),
            ("Папка промтов", self.prompt_var.get()),
            ("Папка контекста", self.context_var.get()),
            ("Папка результатов", self.output_var.get()),
        ]

        # Добавляем папку Whisper если она указана
        whisper_folder = self.whisper_output_var.get()
        if whisper_folder:
            folders.append(("Папка для JSON Whisper", whisper_folder))

        created = []
        errors = []

        for name, path in folders:
            if path:  # Пропускаем пустые пути
                try:
                    os.makedirs(path, exist_ok=True)
                    created.append(name)
                except Exception as e:
                    errors.append(f"{name}: {str(e)}")

        # Показываем результат
        if created and not errors:
            self.set_status(f"Создано папок: {len(created)}", "success")
        elif errors:
            self.set_status(f"Ошибки: {', '.join(errors)}", "error")
        else:
            self.set_status("Все папки уже существуют", "info")

    def set_status(self, message, status_type="info"):
        """Устанавливает статусное сообщение."""
        self.status_var.set(message)

        colors = {
            "info": "black",
            "success": "green",
            "warning": "orange",
            "error": "red",
        }

        color = colors.get(status_type, "black")
        self.status_label.config(foreground=color)

    def get_values(self):
        """Возвращает текущие значения настроек."""
        return {
            "recordings_folder": self.recordings_var.get().strip(),
            "default_audio_folder": self.recordings_var.get().strip()
            or self.config.get("default_audio_folder", str(Path.home())),
            "default_prompt_folder": self.prompt_var.get().strip(),
            "default_output_folder": self.output_var.get().strip(),
            "whisper_output_folder": self.whisper_output_var.get().strip(),
        }


class AudioTab:
    """Вкладка настроек аудиоустройств."""

    def __init__(self, parent, config):
        self.parent = parent
        self.config = config

        self.create_widgets()
        self.load_current_values()
        self.refresh_devices()

    def create_widgets(self):
        """Создает все элементы управления вкладки."""
        row = 0

        # Выходное устройство (системный звук)
        ttk.Label(self.parent, text="Выходное устройство (воспроизведение):").grid(
            row=row, column=0, sticky="w", padx=5, pady=5
        )

        self.output_var = tk.StringVar()
        self.output_combo = ttk.Combobox(
            self.parent,
            textvariable=self.output_var,
            values=[],  # Будет заполнено через refresh_devices
            state="readonly",
            width=50,
        )
        self.output_combo.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        self.output_combo.bind("<<ComboboxSelected>>", self.on_output_selected)
        row += 1

        # Входное устройство (микрофон)
        ttk.Label(self.parent, text="Входное устройство (микрофон):").grid(
            row=row, column=0, sticky="w", padx=5, pady=5
        )

        self.input_var = tk.StringVar()
        self.input_combo = ttk.Combobox(
            self.parent,
            textvariable=self.input_var,
            values=[],  # Будет заполнено через refresh_devices
            state="readonly",
            width=50,
        )
        self.input_combo.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        self.input_combo.bind("<<ComboboxSelected>>", self.on_input_selected)
        row += 1

        # Кнопка обновления списка устройств
        self.refresh_btn = ttk.Button(
            self.parent, text="Обновить список устройств", command=self.refresh_devices
        )
        self.refresh_btn.grid(row=row, column=0, padx=5, pady=10)

        # Чекбокс автообновления
        self.auto_refresh_var = tk.BooleanVar(value=False)
        self.auto_refresh_cb = ttk.Checkbutton(
            self.parent,
            text="Автообновление списка устройств",
            variable=self.auto_refresh_var,
        )
        self.auto_refresh_cb.grid(row=row, column=1, sticky="w", padx=5, pady=10)
        row += 1

        # Панель управления профилями удалена

        # Статусная строка
        self.status_var = tk.StringVar(value="Готово")
        self.status_label = ttk.Label(self.parent, textvariable=self.status_var)
        self.status_label.grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=5
        )

        # Настройка весов колонок
        self.parent.grid_columnconfigure(1, weight=1)

    def load_current_values(self):
        """Загружает текущие значения из конфигурации."""
        # Загружаем сохраненные устройства
        self.output_var.set(self.config.get("audio_output_device", ""))
        self.input_var.set(self.config.get("audio_input_device", ""))
        self.auto_refresh_var.set(self.config.get("audio_auto_refresh_devices", True))

    def refresh_devices(self):
        """Обновляет списки доступных аудиоустройств."""
        try:
            # Импортируем AudioRecorder здесь, чтобы избежать циклических зависимостей
            from audio_recorder import AudioRecorder

            # Создаем временный экземпляр AudioRecorder для получения устройств
            recorder = AudioRecorder(lambda: None, self.config)
            devices_info = recorder.get_available_devices(force_refresh=True)

            output_devices = devices_info.get("output_devices", [])
            input_devices = devices_info.get("input_devices", [])

            # Формируем списки для комбобоксов
            output_names = [
                f"{d['name']} (индекс {d['index']})" for d in output_devices
            ]
            input_names = [f"{d['name']} (индекс {d['index']})" for d in input_devices]

            # Обновляем комбобоксы
            self.output_combo["values"] = output_names
            self.input_combo["values"] = input_names

            # Выбираем устройства по умолчанию если не выбраны
            default_output_idx = devices_info.get("default_output_index")
            default_input_idx = devices_info.get("default_input_index")

            if not self.output_var.get() and default_output_idx is not None:
                for d in output_devices:
                    if d["index"] == default_output_idx:
                        self.output_var.set(f"{d['name']} (индекс {d['index']})")
                        break

            if not self.input_var.get() and default_input_idx is not None:
                for d in input_devices:
                    if d["index"] == default_input_idx:
                        self.input_var.set(f"{d['name']} (индекс {d['index']})")
                        break

            self.set_status(
                f"Найдено {len(output_devices)} выходных и {len(input_devices)} входных устройств",
                "success",
            )

        except Exception as e:
            self.set_status(f"Ошибка при обновлении устройств: {str(e)}", "error")

    def check_devices_available(self):
        """
        Проверяет доступность аудиоустройств (микрофона и выходных устройств).
        Возвращает словарь с информацией о доступности.
        """
        try:
            from audio_recorder import AudioRecorder

            # Создаем временный экземпляр AudioRecorder для проверки
            recorder = AudioRecorder(lambda: None, self.config)
            devices_info = recorder.check_devices_available()

            # Формируем понятный результат
            result = {
                "microphone_available": devices_info.get("microphone_available", False),
                "loopback_available": devices_info.get("loopback_available", False),
                "wasapi_available": devices_info.get("wasapi_available", False),
                "microphone_count": len(devices_info.get("microphone_devices", [])),
                "loopback_count": len(devices_info.get("loopback_devices", [])),
                "default_microphone": devices_info.get("default_microphone"),
                "default_speakers": devices_info.get("default_speakers"),
                "errors": devices_info.get("errors", []),
            }

            # Логируем результат
            status_msg = []
            if result["microphone_available"]:
                status_msg.append(f"Микрофоны: {result['microphone_count']} доступно")
            else:
                status_msg.append("Микрофоны: недоступны")

            if result["loopback_available"]:
                status_msg.append(
                    f"Loopback устройств: {result['loopback_count']} доступно"
                )
            else:
                status_msg.append("Loopback устройства: недоступны")

            self.set_status("; ".join(status_msg), "info")

            return result

        except Exception as e:
            error_msg = f"Ошибка проверки устройств: {str(e)}"
            self.set_status(error_msg, "error")
            return {
                "microphone_available": False,
                "loopback_available": False,
                "wasapi_available": False,
                "microphone_count": 0,
                "loopback_count": 0,
                "errors": [error_msg],
            }

    def on_output_selected(self, event):
        """Обработчик выбора выходного устройства."""
        device_name = self.output_var.get()
        self.set_status(f"Выбрано выходное устройство: {device_name}", "info")

    def on_input_selected(self, event):
        """Обработчик выбора входного устройства."""
        device_name = self.input_var.get()
        self.set_status(f"Выбрано входное устройство: {device_name}", "info")

    def set_status(self, message, status_type="info"):
        """Устанавливает статусное сообщение."""
        self.status_var.set(message)

        colors = {
            "info": "black",
            "success": "green",
            "warning": "orange",
            "error": "red",
        }

        color = colors.get(status_type, "black")
        self.status_label.config(foreground=color)

    def _parse_device_string(self, device_str):
        """Извлекает имя и индекс устройства из строки формата 'Имя (индекс X)'."""
        if not device_str:
            return "", -1

        # Пытаемся найти индекс в скобках
        import re

        match = re.search(r"\(индекс\s+(\d+)\)", device_str)
        if match:
            index = int(match.group(1))
            # Извлекаем имя без индекса
            name = re.sub(r"\s*\(индекс\s+\d+\)", "", device_str).strip()
            return name, index
        else:
            # Если индекс не найден, возвращаем всю строку как имя
            return device_str, -1

    def _get_device_index(self, device_str, device_type="output"):
        """Возвращает индекс устройства по строке выбора."""
        name, index = self._parse_device_string(device_str)
        if index >= 0:
            return index

        # Если индекс не найден, пытаемся найти устройство по имени в кэше
        try:
            from audio_recorder import AudioRecorder

            recorder = AudioRecorder(lambda: None, self.config)
            devices_info = recorder.get_available_devices(force_refresh=False)
            devices = devices_info.get(f"{device_type}_devices", [])
            for d in devices:
                if d["name"] == name:
                    return d["index"]
        except:
            pass
        return -1

    def get_values(self):
        """Возвращает текущие значения настроек."""
        output_str = self.output_var.get()
        input_str = self.input_var.get()

        output_name, output_idx = self._parse_device_string(output_str)
        input_name, input_idx = self._parse_device_string(input_str)

        # Если индекс не найден, пытаемся получить его
        if output_idx < 0:
            output_idx = self._get_device_index(output_str, "output")
        if input_idx < 0:
            input_idx = self._get_device_index(input_str, "input")

        return {
            "audio_output_device": output_name,
            "audio_input_device": input_name,
            "audio_output_device_id": output_idx,
            "audio_input_device_id": input_idx,
            "audio_auto_refresh_devices": self.auto_refresh_var.get(),
        }

    def cleanup(self):
        """Очистка ресурсов AudioTab (заглушка, функциональность удалена)."""
        pass


class SettingsDialog:
    """Модальное окно настроек с вкладками."""

    def __init__(self, parent, config):
        self.parent = parent
        self.config = config.copy()
        self.api_key = get_api_key()

        self.dialog = tk.Toplevel(parent)
        self.setup_dialog()
        self.create_notebook()
        self.create_buttons()

    def setup_dialog(self):
        self.dialog.title("Настройки MMAssistant")
        self.dialog.geometry("700x400")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        # Обработчик закрытия окна через крестик
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_window_close)

    def create_notebook(self):
        self.notebook = ttk.Notebook(self.dialog)

        # Создание вкладок
        self.llm_frame = ttk.Frame(self.notebook)
        self.whisper_frame = ttk.Frame(self.notebook)
        self.folders_frame = ttk.Frame(self.notebook)
        self.audio_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.llm_frame, text="LLM")
        self.notebook.add(self.whisper_frame, text="Whisper")
        self.notebook.add(self.folders_frame, text="Папки")
        self.notebook.add(self.audio_frame, text="Аудио")

        # Инициализация вкладок
        self.llm_tab = LLMTab(self.llm_frame, self.config, self.api_key)
        self.whisper_tab = WhisperTab(self.whisper_frame, self.config)
        self.folders_tab = FoldersTab(self.folders_frame, self.config)
        self.audio_tab = AudioTab(self.audio_frame, self.config)

        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_buttons(self):
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Сохранить", command=self.on_save).pack(
            side=tk.RIGHT, padx=5
        )
        ttk.Button(button_frame, text="Отмена", command=self.on_cancel).pack(
            side=tk.RIGHT, padx=5
        )

    def on_save(self):
        """Сохраняет все настройки."""
        try:
            # Сохраняем настройки LLM
            llm_values = self.llm_tab.get_values()
            new_api_key = self.llm_tab.get_api_key()

            # Сохраняем ключ если он изменился
            if new_api_key is not None:
                set_api_key(new_api_key)

            # Сохраняем остальные настройки
            whisper_values = self.whisper_tab.get_values()
            folders_values = self.folders_tab.get_values()
            audio_values = self.audio_tab.get_values()

            # Обновление конфигурации
            self.config.update(llm_values)
            self.config.update(whisper_values)
            self.config.update(folders_values)
            self.config.update(audio_values)

            # Сохранение в файл
            save_config(self.config)

            self.dialog.destroy()

        except Exception as e:
            messagebox.showerror(
                "Ошибка сохранения", f"Не удалось сохранить настройки: {e}"
            )

    def on_cancel(self):
        self.cleanup_tabs()
        self.dialog.destroy()

    def cleanup_tabs(self):
        """Очистка ресурсов всех вкладок перед закрытием диалога."""
        # Очищаем ресурсы AudioTab (останавливаем мониторинг, уничтожаем VU-метр)
        if hasattr(self, "audio_tab"):
            try:
                self.audio_tab.cleanup()
            except Exception as e:
                print(f"Ошибка при очистке AudioTab: {e}")

        # Дополнительно можно очистить другие вкладки при необходимости
        # if hasattr(self, 'llm_tab'):
        #     self.llm_tab.cleanup()

    def on_window_close(self):
        """Обработчик закрытия окна через крестик."""
        self.cleanup_tabs()
        self.dialog.destroy()
