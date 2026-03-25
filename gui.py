import os
import json
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from datetime import datetime
from pathlib import Path

# Импорт модуля записи аудио
from audio_recorder import AudioRecorder

# Импорт модуля конфигурации
from config_manager import (
    load_config,
    save_config,
    has_api_key,
    migrate_old_config,
)

# Импорт модуля настроек
from settings_module import SettingsDialog

# Импорт модуля обработки запросов
from request_processor import (
    transcribe_audio,
    extract_text_from_whisper_result,
    query_llm,
    remove_think_tags,
    save_md_file,
    save_whisper_result,
)


# ==================== ГРАФИЧЕСКИЙ ИНТЕРФЕЙС ====================
class App:
    PLACEHOLDER_TEXT = "Выбери промт из списка"
    CONTEXT_PLACEHOLDER_TEXT = "(Опционально) выбери дополнительный контекст"
    AUDIO_PLACEHOLDER_TEXT = "Выбери файл записи"

    def __init__(self, root):
        self.root = root
        self.root.title("Meeting Minutes Assistant")
        self.root.geometry("800x780")
        self.root.resizable(False, False)

        self.config = load_config()
        self.audio_path = tk.StringVar(value=self.AUDIO_PLACEHOLDER_TEXT)
        self.prompt_path = tk.StringVar(value=self.PLACEHOLDER_TEXT)
        self.context_path = tk.StringVar(value=self.CONTEXT_PLACEHOLDER_TEXT)
        self.output_folder = tk.StringVar(
            value=self.config.get("default_output_folder", "")
        )

        # Фактический полный путь к аудиофайлу (для обработки)
        self.actual_audio_path = ""
        # Фактический полный путь к файлу промпта (для обработки)
        self.actual_prompt_path = ""
        # Фактический полный путь к файлу контекста (для обработки)
        self.actual_context_path = ""

        self.recorder = AudioRecorder(self.on_recording_finished, self.config)

        # Атрибуты для выпадающего списка файлов промтов
        self.prompt_files = []  # список файлов промтов
        self.current_prompt_folder = ""  # текущая папка промтов
        # Атрибуты для выпадающего списка файлов контекста
        self.context_files = []  # список файлов контекста
        self.current_context_folder = ""  # текущая папка контекста
        # Атрибуты для выпадающего списка аудиофайлов
        self.audio_files = []  # список аудиофайлов и JSON
        self.current_audio_folder = ""  # текущая папка аудио

        self.audio_path.trace("w", self._check_ready)
        self.prompt_path.trace("w", self._check_ready)
        self.context_path.trace("w", self._check_ready)
        self.output_folder.trace("w", self._check_ready)

        self.create_widgets()

        # Заполняем выпадающий список файлов промтов
        self.update_prompt_list()
        # Заполняем выпадающий список файлов контекста
        self.update_context_list()
        # Заполняем выпадающий список аудиофайлов
        self.update_audio_list()

        # Проверяем и выполняем миграцию если нужно
        self.migrate_if_needed()

        # Проверяем наличие API ключа
        self.check_api_key()

    def migrate_if_needed(self):
        """Выполняет миграцию старой конфигурации."""
        try:
            if migrate_old_config():
                self.log(
                    "Старый API ключ удален из config.json. Введите ключ в настройках."
                )
        except Exception as e:
            self.log(f"Ошибка миграции: {e}")

    def check_api_key(self):
        """Проверяет наличие API ключа и показывает предупреждение если нужно."""
        try:
            if not has_api_key():
                # Показываем информационное сообщение
                messagebox.showinfo(
                    "Настройка API ключа",
                    "API ключ не настроен.\n\n"
                    "Для работы с LLM необходимо:\n"
                    "1. Нажмите кнопку 'Настройки'\n"
                    "2. Перейдите на вкладку 'LLM'\n"
                    "3. Введите ваш API ключ\n"
                    "4. Нажмите 'Сохранить'\n\n"
                    "Ключ будет безопасно сохранен в Windows Credential Manager.",
                )
        except RuntimeError as e:
            # Ошибка keyring - критическая
            messagebox.showerror(
                "Ошибка keyring",
                f"Не удалось получить доступ к keyring:\n{str(e)}\n\n"
                "Приложение не сможет работать с LLM API.\n"
                "Проверьте настройки системы и права доступа.",
            )

    def set_audio_path(self, full_path):
        """Устанавливает полный путь к аудиофайлу и обновляет отображение в комбобоксе.

        Args:
            full_path: Полный путь к аудиофайлу или JSON
        """
        self.actual_audio_path = full_path
        if full_path and full_path != self.AUDIO_PLACEHOLDER_TEXT:
            # В комбобокс устанавливаем только имя файла
            file_name = os.path.basename(full_path)
            self.audio_path.set(file_name)
        else:
            new_value = full_path if full_path else self.AUDIO_PLACEHOLDER_TEXT
            self.audio_path.set(new_value)

    def set_prompt_path(self, full_path):
        """Устанавливает полный путь к файлу промпта и обновляет отображение в комбобоксе.

        Args:
            full_path: Полный путь к файлу промпта
        """
        self.actual_prompt_path = full_path
        if full_path and full_path != self.PLACEHOLDER_TEXT:
            # В комбобокс устанавливаем только имя файла
            file_name = os.path.basename(full_path)
            self.prompt_path.set(file_name)
        else:
            new_value = full_path if full_path else self.PLACEHOLDER_TEXT
            self.prompt_path.set(new_value)

    def set_context_path(self, full_path):
        """Устанавливает полный путь к файлу контекста и обновляет отображение в комбобоксе.

        Args:
            full_path: Полный путь к файлу контекста
        """
        self.actual_context_path = full_path
        if full_path and full_path != self.CONTEXT_PLACEHOLDER_TEXT:
            # В комбобокс устанавливаем только имя файла
            file_name = os.path.basename(full_path)
            self.context_path.set(file_name)
        else:
            new_value = full_path if full_path else self.CONTEXT_PLACEHOLDER_TEXT
            self.context_path.set(new_value)

    def clear_input_fields(self):
        """Очищает поля ввода аудио, промпта и контекста после успешной обработки.

        Устанавливает значения placeholder'ов в StringVar и очищает фактические пути.
        Папка сохранения результата (self.output_folder) не очищается.
        """
        # Устанавливаем placeholder'ы
        self.audio_path.set(self.AUDIO_PLACEHOLDER_TEXT)
        self.prompt_path.set(self.PLACEHOLDER_TEXT)
        self.context_path.set(self.CONTEXT_PLACEHOLDER_TEXT)

        # Очищаем фактические пути через вызовы существующих методов
        self.set_audio_path("")
        self.set_prompt_path("")
        self.set_context_path("")

        # Папка сохранения не очищается (self.output_folder остается как есть)
        # Опционально: обновляем списки файлов в комбобоксах
        self.update_audio_list()
        self.update_prompt_list()
        self.update_context_list()

    def get_audio_path(self):
        """Возвращает полный путь к аудиофайлу для обработки."""
        # Если actual_audio_path не установлен, но в audio_path есть значение (имя файла),
        # пытаемся построить полный путь из текущей папки
        if (
            self.actual_audio_path
            and self.actual_audio_path != self.AUDIO_PLACEHOLDER_TEXT
        ):
            return self.actual_audio_path
        # Если в audio_path только имя файла (не полный путь) и есть текущая папка
        current_value = self.audio_path.get()
        if (
            current_value
            and current_value != self.AUDIO_PLACEHOLDER_TEXT
            and self.current_audio_folder
            and not os.path.isabs(current_value)
        ):
            return os.path.join(self.current_audio_folder, current_value)
        # Иначе возвращаем значение из audio_path (может быть полным путем или placeholder)
        return current_value

    def get_prompt_path(self):
        """Возвращает полный путь к файлу промпта для обработки."""
        # Если actual_prompt_path не установлен, но в prompt_path есть значение (имя файла),
        # пытаемся построить полный путь из текущей папки
        if self.actual_prompt_path and self.actual_prompt_path != self.PLACEHOLDER_TEXT:
            return self.actual_prompt_path
        # Если в prompt_path только имя файла (не полный путь) и есть текущая папка
        current_value = self.prompt_path.get()
        if (
            current_value
            and current_value != self.PLACEHOLDER_TEXT
            and self.current_prompt_folder
            and not os.path.isabs(current_value)
        ):
            return os.path.join(self.current_prompt_folder, current_value)
        # Иначе возвращаем значение из prompt_path (может быть полным путем или placeholder)
        return current_value

    def get_context_path(self):
        """Возвращает полный путь к файлу контекста для обработки."""
        # Если actual_context_path не установлен, но в context_path есть значение (имя файла),
        # пытаемся построить полный путь из текущей папки
        if (
            self.actual_context_path
            and self.actual_context_path != self.CONTEXT_PLACEHOLDER_TEXT
        ):
            return self.actual_context_path
        # Если в context_path только имя файла (не полный путь) и есть текущая папка
        current_value = self.context_path.get()
        if (
            current_value
            and current_value != self.CONTEXT_PLACEHOLDER_TEXT
            and self.current_context_folder
            and not os.path.isabs(current_value)
        ):
            return os.path.join(self.current_context_folder, current_value)
        # Иначе возвращаем значение из context_path (может быть полным путем или placeholder)
        return current_value

    def open_settings(self):
        """Открывает окно настроек."""
        dialog = SettingsDialog(self.root, self.config)
        self.root.wait_window(dialog.dialog)

        # Обновляем конфигурацию
        self.config = load_config()
        self.log("Настройки обновлены")

    def _check_ready(self, *args):
        prompt_text = self.prompt_path.get().strip()
        context_text = self.context_path.get().strip()
        audio_text = self.audio_path.get().strip()
        # Игнорируем placeholder при проверке
        if prompt_text == self.PLACEHOLDER_TEXT:
            prompt_text = ""
        if context_text == self.CONTEXT_PLACEHOLDER_TEXT:
            context_text = ""
        if audio_text == self.AUDIO_PLACEHOLDER_TEXT:
            audio_text = ""
        if audio_text and prompt_text and self.output_folder.get().strip():
            self.process_btn.config(state="normal")
        else:
            self.process_btn.config(state="disabled")
        if audio_text:
            self.whisper_only_btn.config(state="normal")
        else:
            self.whisper_only_btn.config(state="disabled")

    def create_widgets(self):
        # Большая кнопка записи вверху
        self.record_btn = tk.Button(
            self.root,
            text="НАЧАТЬ ЗАПИСЬ",
            command=self.toggle_recording,
            bg="lightcoral",
            font=("Arial", 14, "bold"),
            height=2,
        )
        self.record_btn.grid(
            row=0, column=0, columnspan=3, padx=20, pady=(10, 5), sticky="ew"
        )

        self.record_status = tk.Label(self.root, text="", fg="blue", font=("Arial", 10))
        self.record_status.grid(row=1, column=0, columnspan=3, pady=5)

        row = 2
        tk.Button(
            self.root,
            text="ВЫБРАТЬ АУДИОФАЙЛ / РАСШИФРОВКУ",
            command=self.browse_audio,
            font=("Arial", 10),
            width=40,
        ).grid(row=row, column=0, padx=(10, 5), pady=5, sticky="w")
        self.audio_combo = ttk.Combobox(
            self.root, textvariable=self.audio_path, width=48
        )
        self.audio_combo.grid(row=row, column=1, padx=(5, 5), pady=5, sticky="ew")
        self.audio_combo.bind("<<ComboboxSelected>>", self.on_audio_selected)
        self.audio_combo.bind("<FocusIn>", lambda e: self.update_audio_list())
        row += 1

        tk.Button(
            self.root,
            text="ВЫБРАТЬ ФАЙЛ ПРОМПТА",
            command=self.browse_prompt,
            font=("Arial", 10),
            width=40,
        ).grid(row=row, column=0, padx=(10, 5), pady=5, sticky="w")
        # Выпадающий список файлов промтов
        self.prompt_combo = ttk.Combobox(
            self.root, textvariable=self.prompt_path, width=48
        )
        self.prompt_combo.grid(row=row, column=1, padx=(5, 5), pady=5, sticky="ew")
        self.prompt_combo.bind("<<ComboboxSelected>>", self.on_prompt_selected)
        self.prompt_combo.bind("<FocusIn>", lambda e: self.update_prompt_list())
        row += 1

        tk.Button(
            self.root,
            text="КОНТЕКСТ К ПРОМТУ",
            command=self.browse_context,
            font=("Arial", 10),
            width=40,
        ).grid(row=row, column=0, padx=(10, 5), pady=5, sticky="w")
        # Выпадающий список файлов контекста
        self.context_combo = ttk.Combobox(
            self.root, textvariable=self.context_path, width=48
        )
        self.context_combo.grid(row=row, column=1, padx=(5, 5), pady=5, sticky="ew")
        self.context_combo.bind("<<ComboboxSelected>>", self.on_context_selected)
        self.context_combo.bind("<FocusIn>", lambda e: self.update_context_list())
        row += 1

        tk.Button(
            self.root,
            text="ВЫБРАТЬ ПАПКУ СОХРАНЕНИЯ",
            command=self.browse_output_folder,
            font=("Arial", 10),
            width=40,
        ).grid(row=row, column=0, padx=(10, 5), pady=5, sticky="w")
        self.entry_output = tk.Entry(
            self.root, textvariable=self.output_folder, width=50
        )
        self.entry_output.grid(
            row=row, column=1, columnspan=2, padx=5, pady=5, sticky="ew"
        )
        row += 1

        # Лог
        log_frame = tk.LabelFrame(
            self.root, text="Лог", font=("Arial", 10, "bold"), padx=10, pady=10
        )
        log_frame.grid(row=row, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.log_area = scrolledtext.ScrolledText(
            log_frame, width=85, height=10, state="normal", font=("Courier", 9)
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)
        row += 1

        # Крупные кнопки внизу
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=row, column=0, columnspan=3, pady=(10, 20))

        self.process_btn = tk.Button(
            button_frame,
            text="ОБРАБОТАТЬ",
            command=self.start_processing,
            bg="lightblue",
            font=("Arial", 12, "bold"),
            width=12,
            height=3,
            state="disabled",
        )
        self.process_btn.pack(side=tk.LEFT, padx=15)

        self.whisper_only_btn = tk.Button(
            button_frame,
            text="Перевести\nв текст",
            command=self.start_whisper_only,
            font=("Arial", 12, "bold"),
            width=12,
            height=3,
            state="disabled",
        )
        self.whisper_only_btn.pack(side=tk.LEFT, padx=15)

        tk.Button(
            button_frame,
            text="НАСТРОЙКИ",
            command=self.open_settings,
            font=("Arial", 11),
            width=12,
            height=3,
        ).pack(side=tk.LEFT, padx=15)

        tk.Button(
            button_frame,
            text="ВЫХОД",
            command=self.root.quit,
            font=("Arial", 12, "bold"),
            width=12,
            height=3,
        ).pack(side=tk.LEFT, padx=15)

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(row - 1, weight=1)

    def browse_audio(self):
        filename = filedialog.askopenfilename(
            title="Выберите аудиофайл или JSON расшифровку",
            initialdir=self.config.get("default_audio_folder", ""),
            filetypes=[
                ("Аудио файлы", "*.mp3 *.wav *.m4a *.ogg *.flac"),
                ("JSON расшифровки Whisper", "*.json"),
                ("Все файлы", "*.*"),
            ],
        )
        if filename:
            self.set_audio_path(filename)
            self.config["default_audio_folder"] = os.path.dirname(filename)
            # Обновляем список файлов в комбобоксе
            self.update_audio_list()

    def browse_prompt(self):
        filename = filedialog.askopenfilename(
            title="Выберите файл с промптом",
            initialdir=self.config.get("default_prompt_folder", ""),
            filetypes=[("Текстовые файлы", "*.txt *.md"), ("Все файлы", "*.*")],
        )
        if filename:
            normalized_path = os.path.normpath(filename)
            self.set_prompt_path(normalized_path)
            self.config["default_prompt_folder"] = os.path.dirname(normalized_path)
            # Обновляем список файлов в выпадающем списке
            self.update_prompt_list()

    def browse_context(self):
        filename = filedialog.askopenfilename(
            title="Выберите файл контекста",
            initialdir=self.config.get("default_prompt_folder", ""),
            filetypes=[("Текстовые файлы", "*.txt *.md"), ("Все файлы", "*.*")],
        )
        if filename:
            normalized_path = os.path.normpath(filename)
            self.set_context_path(normalized_path)
            # Обновляем список файлов в выпадающем списке
            self.update_context_list()

    def update_prompt_list(self):
        """Сканирует папку default_prompt_folder из конфигурации, находит файлы .txt и .md,
        сортирует и обновляет значения в комбобоксе."""
        folder = self.config.get("default_prompt_folder", "")
        if not folder or not os.path.exists(folder):
            self.prompt_files = []
            self.current_prompt_folder = ""
            self.prompt_combo["values"] = []
            return

        self.current_prompt_folder = folder
        files = []
        for ext in ("*.txt", "*.md"):
            files.extend(Path(folder).glob(ext))

        # Получаем только имена файлов, сортируем по алфавиту
        file_names = sorted([f.name for f in files])
        self.prompt_files = file_names
        self.prompt_combo["values"] = file_names
        # Если в папке есть файлы, можно установить первый как текущее значение?
        # Не будем автоматически устанавливать, чтобы не перезаписать выбранный файл.

    def on_prompt_selected(self, event):
        """Обработчик выбора файла из выпадающего списка."""
        selected_file = self.prompt_combo.get()
        if not selected_file:
            return
        if self.current_prompt_folder:
            full_path = os.path.join(self.current_prompt_folder, selected_file)
            normalized_path = os.path.normpath(full_path)
            self.set_prompt_path(normalized_path)
        # Примечание: значение в self.prompt_path уже установлено через textvariable,
        # но мы также можем обновить конфигурацию папки по умолчанию?
        # Не будем менять default_prompt_folder, так как папка уже установлена.

    def update_context_list(self):
        """Сканирует папку default_prompt_folder из конфигурации, находит файлы .txt и .md,
        сортирует и обновляет значения в комбобоксе контекста."""
        folder = self.config.get("default_prompt_folder", "")
        if not folder or not os.path.exists(folder):
            self.context_files = []
            self.current_context_folder = ""
            self.context_combo["values"] = []
            return

        self.current_context_folder = folder
        files = []
        for ext in ("*.txt", "*.md"):
            files.extend(Path(folder).glob(ext))

        # Получаем только имена файлов, сортируем по алфавиту
        file_names = sorted([f.name for f in files])
        self.context_files = file_names
        self.context_combo["values"] = file_names
        # Если в папке есть файлы, можно установить первый как текущее значение?
        # Не будем автоматически устанавливать, чтобы не перезаписать выбранный файл.

    def is_whisper_json(self, filepath):
        """Проверяет, является ли JSON файл результатом Whisper.
        Соответствует логике валидации в request_processor.py."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Проверяем наличие ожидаемых полей в результате Whisper
            if isinstance(data, dict):
                # Whisper результат может содержать 'text' или 'segments'
                if "text" in data or "segments" in data:
                    return True
            return False
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return False

    def update_audio_list(self):
        """Сканирует папку default_audio_folder из конфигурации, находит аудиофайлы и JSON,
        сортирует и обновляет значения в комбобоксе аудио."""
        folder = self.config.get("default_audio_folder", "")
        if not folder or not os.path.exists(folder):
            self.audio_files = []
            self.current_audio_folder = ""
            self.audio_combo["values"] = []
            return

        self.current_audio_folder = folder
        files = []
        # Расширения аудиофайлов
        audio_exts = ("*.mp3", "*.wav", "*.m4a", "*.ogg", "*.flac")
        for ext in audio_exts:
            files.extend(Path(folder).glob(ext))
        # JSON файлы
        json_files = list(Path(folder).glob("*.json"))
        # Фильтруем только валидные Whisper JSON
        valid_json_files = [f for f in json_files if self.is_whisper_json(f)]
        files.extend(valid_json_files)

        # Получаем только имена файлов, сортируем по алфавиту
        file_names = sorted([f.name for f in files])
        self.audio_files = file_names
        self.audio_combo["values"] = file_names
        # Если в папке есть файлы, можно установить первый как текущее значение?
        # Не будем автоматически устанавливать, чтобы не перезаписать выбранный файл.

    def on_audio_selected(self, event):
        """Обработчик выбора аудиофайла или JSON из выпадающего списка."""
        selected_file = self.audio_combo.get()
        if not selected_file:
            return
        if self.current_audio_folder:
            full_path = os.path.join(self.current_audio_folder, selected_file)
            self.set_audio_path(full_path)

    def on_context_selected(self, event):
        """Обработчик выбора файла контекста из выпадающего списка."""
        selected_file = self.context_combo.get()
        if not selected_file:
            return
        if self.current_context_folder:
            full_path = os.path.join(self.current_context_folder, selected_file)
            normalized_path = os.path.normpath(full_path)
            self.set_context_path(normalized_path)

    def browse_output_folder(self):
        folder = filedialog.askdirectory(
            title="Выберите папку для сохранения",
            initialdir=self.output_folder.get()
            or self.config.get("default_output_folder", ""),
        )
        if folder:
            self.output_folder.set(folder)
            self.config["default_output_folder"] = folder

    def save_current_config(self):
        current_on_disk = load_config()
        safe_keys = [
            "recordings_folder",
            "whisper_output_folder",
            "default_audio_folder",
            "default_prompt_folder",
            "default_output_folder",
        ]
        for key in safe_keys:
            if key in self.config:
                current_on_disk[key] = self.config[key]
        save_config(current_on_disk)
        self.config = current_on_disk
        self.log("Конфигурация сохранена в файл.")

    def log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.root.update()

    def toggle_recording(self):
        if not self.recorder.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def _get_whisper_json_path(self, audio_path):
        folder = self.config.get("whisper_output_folder", "")
        if not folder:
            folder = os.path.dirname(audio_path)
        base_name = Path(audio_path).stem
        return os.path.join(folder, f"{base_name}.json")

    def start_recording(self):
        rec_folder = self.config.get("recordings_folder", "")
        if not rec_folder or not os.path.exists(rec_folder):
            folder = filedialog.askdirectory(
                title="Выберите папку для сохранения аудиозаписей",
                initialdir=self.config.get("default_audio_folder") or str(Path.home()),
            )
            if not folder:
                self.log("Запись отменена: не выбрана папка.")
                return
            rec_folder = folder
            self.config["recordings_folder"] = rec_folder
            save_config(self.config)
            self.log(f"Папка для записей установлена: {rec_folder}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.mp3"
        self.current_recording_path = os.path.join(rec_folder, filename)

        self.record_btn.config(text="Остановить запись", bg="lightgreen")
        self.record_status.config(text="Идёт запись...", fg="green")
        self.log(f"Начало записи в {self.current_recording_path}")

        self.recorder.start_recording(self.current_recording_path)

    def stop_recording(self):
        self.record_btn.config(text="Начать запись", bg="lightcoral")
        self.record_status.config(text="Остановка...", fg="blue")
        self.log("Остановка записи...")
        self.recorder.stop_recording()

    def on_recording_finished(self, success, filepath, error_msg):
        if success:
            self.root.after(0, lambda p=filepath: self.set_audio_path(p))
            self.config["default_audio_folder"] = os.path.dirname(filepath)
            save_config(self.config)
            self.root.after(
                0, lambda: self.record_status.config(text="Запись сохранена", fg="gray")
            )
            self.root.after(0, lambda p=filepath: self.log(f"Запись сохранена: {p}"))
            # Обновляем список файлов в комбобоксе
            self.root.after(0, self.update_audio_list)

        else:
            self.root.after(
                0, lambda: self.record_status.config(text="Ошибка записи", fg="red")
            )
            self.root.after(0, lambda msg=error_msg: self.log(f"Ошибка записи: {msg}"))
            self.root.after(
                0, lambda msg=error_msg: messagebox.showerror("Ошибка записи", msg)
            )

    def start_processing(self):
        audio = self.get_audio_path()
        prompt = self.get_prompt_path()
        out_folder = self.output_folder.get()

        if not audio or not prompt or audio == self.AUDIO_PLACEHOLDER_TEXT:
            messagebox.showerror("Ошибка", "Выберите аудиофайл и файл с промптом.")
            return
        if not out_folder:
            out_folder = self.config.get("default_output_folder", "")
            if not out_folder:
                out_folder = os.path.dirname(audio)
            self.output_folder.set(out_folder)

        self.process_btn.config(state="disabled")
        self.whisper_only_btn.config(state="disabled")
        self.log("Начало обработки...")

        thread = threading.Thread(target=self.process, args=(audio, prompt, out_folder))
        thread.daemon = True
        thread.start()

    def process(self, audio_path, prompt_path, output_folder):
        try:
            self.log("Чтение файла промпта...")
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read()

            context_file = self.get_context_path().strip()
            # Игнорируем placeholder при проверке
            if context_file == self.CONTEXT_PLACEHOLDER_TEXT:
                context_file = ""
            if context_file:
                try:
                    with open(context_file, "r", encoding="utf-8") as cf:
                        context_content = cf.read()
                    prompt_text = prompt_text.replace("{Context}", context_content)
                    self.log("Контекст успешно подставлен в промт.")
                except Exception as e:
                    self.log(
                        f"Ошибка чтения файла контекста: {e}. Промт передан без изменений."
                    )

            json_path = self._get_whisper_json_path(audio_path)
            if os.path.exists(json_path):
                self.log(f"Найден существующий JSON: {json_path}. Использую его.")
                with open(json_path, "r", encoding="utf-8") as f:
                    whisper_result = json.load(f)
            else:
                self.log("Отправка аудио на Whisper ASR...")
                whisper_result = transcribe_audio(audio_path, self.config)
                json_path = save_whisper_result(audio_path, whisper_result, self.config)
                self.log(f"Результат Whisper сохранён в JSON: {json_path}")

            asr_text = extract_text_from_whisper_result(whisper_result)
            if not asr_text:
                self.log("Предупреждение: Whisper вернул пустой текст.")
            else:
                self.log(
                    f"Распознано: {asr_text[:100]}..."
                    if len(asr_text) > 100
                    else f"Распознано: {asr_text}"
                )

            self.log("Отправка запроса в LLM...")
            raw_llm_response = query_llm(prompt_text, asr_text, self.config)
            cleaned_response = remove_think_tags(raw_llm_response)

            base_name = os.path.basename(audio_path)
            saved_path = save_md_file(cleaned_response, output_folder, base_name)
            self.log(f"Результат LLM сохранён в MD: {saved_path}")

            self.config["default_output_folder"] = output_folder
            self.log("Обработка завершена успешно!")
            self.root.after(
                0,
                lambda j=json_path, m=saved_path: messagebox.showinfo(
                    "Готово",
                    f"Whisper JSON: {j}\nLLM MD: {m}",
                ),
            )
            # Очистка полей ввода после успешной обработки
            self.root.after(0, self.clear_input_fields)

        except Exception as e:
            self.log(f"ОШИБКА: {e}")
            self.root.after(0, lambda err=e: messagebox.showerror("Ошибка", str(err)))
        finally:
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
        self.root.after(0, lambda: self.whisper_only_btn.config(state="normal"))

    def start_whisper_only(self):
        audio = self.get_audio_path()
        if not audio or audio == self.AUDIO_PLACEHOLDER_TEXT:
            messagebox.showerror("Ошибка", "Выберите аудиофайл.")
            return

        self.process_btn.config(state="disabled")
        self.whisper_only_btn.config(state="disabled")
        self.log("Начало распознавания (только Whisper)...")

        thread = threading.Thread(target=self.whisper_only_process, args=(audio,))
        thread.daemon = True
        thread.start()

    def whisper_only_process(self, audio_path):
        try:
            json_path = self._get_whisper_json_path(audio_path)
            if os.path.exists(json_path):
                self.log(f"Найден существующий JSON: {json_path}. Использую его.")
                with open(json_path, "r", encoding="utf-8") as f:
                    whisper_result = json.load(f)
                asr_text = extract_text_from_whisper_result(whisper_result)
                self.log(
                    f"Распознано: {asr_text[:100]}..."
                    if len(asr_text) > 100
                    else f"Распознано: {asr_text}"
                )
                self.root.after(
                    0,
                    lambda p=json_path: messagebox.showinfo(
                        "Готово", f"Результат уже существовал:\n{p}"
                    ),
                )
                # Очистка полей ввода после успешной обработки
                self.root.after(0, self.clear_input_fields)
            else:
                self.log("Отправка аудио на Whisper ASR...")
                whisper_result = transcribe_audio(audio_path, self.config)
                asr_text = extract_text_from_whisper_result(whisper_result)
                if not asr_text:
                    self.log("Предупреждение: Whisper вернул пустой текст.")
                else:
                    self.log(
                        f"Распознано: {asr_text[:100]}..."
                        if len(asr_text) > 100
                        else f"Распознано: {asr_text}"
                    )
                json_path = save_whisper_result(audio_path, whisper_result, self.config)
                self.log(f"Результат сохранён в JSON: {json_path}")
                self.root.after(
                    0,
                    lambda p=json_path: messagebox.showinfo(
                        "Готово", f"Результат сохранён в:\n{p}"
                    ),
                )
                # Очистка полей ввода после успешной обработки
                self.root.after(0, self.clear_input_fields)
        except Exception as e:
            self.log(f"ОШИБКА: {e}")
            self.root.after(0, lambda err=e: messagebox.showerror("Ошибка", str(err)))
        finally:
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
            self.root.after(0, lambda: self.whisper_only_btn.config(state="normal"))


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
