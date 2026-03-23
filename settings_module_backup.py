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
        self.status_var.set
