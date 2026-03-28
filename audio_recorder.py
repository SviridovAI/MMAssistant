import os
import threading
import time
import numpy as np
import pyaudiowpatch as pyaudio
import traceback
import lameenc
import samplerate
import logging
from datetime import datetime
from pathlib import Path


# Функции для ресемплинга аудио
def create_resampler(mode="sinc_best", channels=1):
    return samplerate.Resampler(mode, channels=channels)


def resample_audio(data, ratio, resampler=None):
    """Ресемплит аудио данные"""
    return resampler.process(data, ratio)


# ==================== ОКОННЫЕ ФУНКЦИИ ====================
def apply_window(data, window_type="hann", strength=0.3):
    """
    Применяет оконную функцию к данным для сглаживания краев.

    Args:
        data: numpy array с аудиоданными
        window_type: тип окна ("hann", "hamming", "blackman", "tukey")
        strength: сила применения окна (0.0 - 1.0), где 1.0 - полное окно
    """
    n = len(data)
    if n < 2 or strength <= 0:
        return data

    # Создаем окно
    if window_type == "hann":
        # Окно Ханна (менее агрессивное)
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    elif window_type == "hamming":
        # Окно Хемминга
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    elif window_type == "blackman":
        # Окно Блэкмана
        window = (
            0.42
            - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
            + 0.08 * np.cos(4 * np.pi * np.arange(n) / (n - 1))
        )
    elif window_type == "tukey":
        # Окно Тьюки (более мягкое, затухает только края)
        alpha = 0.5  # Параметр Тьюки (0.5 - умеренное затухание)
        window = np.ones(n)
        r = int(alpha * (n - 1) / 2)
        if r > 0:
            window[:r] = 0.5 * (1 + np.cos(np.pi * (np.arange(r) / r - 1)))
            window[-r:] = 0.5 * (1 + np.cos(np.pi * (np.arange(r) / r)))
    else:
        # Прямоугольное окно (без изменений)
        return data

    # Применяем силу окна: смешиваем с прямоугольным окном
    if strength < 1.0:
        window = window * strength + (1 - strength)

    return data * window


# ==================== НОРМАЛИЗАЦИЯ АУДИО ====================
def normalize_audio(data, target_level_db=-3.0, eps=1e-8, max_gain_db=40.0):
    """
    Нормализует аудиосигнал до заданного уровня в dBFS.

    Args:
        data: numpy array с аудиоданными (float32)
        target_level_db: целевой уровень в dBFS (по умолчанию -3 dB)
        eps: маленькое значение для избежания деления на ноль
        max_gain_db: максимальное усиление в dB (по умолчанию +40 dB)

    Returns:
        Нормализованный аудиосигнал
    """
    if len(data) == 0:
        return data

    # Вычисляем RMS (среднеквадратичное значение)
    rms = np.sqrt(np.mean(data**2) + eps)

    # Преобразуем RMS в dBFS
    if rms > eps:
        current_level_db = 20 * np.log10(rms)

        # Вычисляем коэффициент усиления
        gain_db = target_level_db - current_level_db

        # Ограничиваем максимальное усиление
        max_gain_linear = 10 ** (max_gain_db / 20)
        gain_linear = 10 ** (gain_db / 20)
        gain_linear = np.clip(gain_linear, 0, max_gain_linear)

        return data * gain_linear
    else:
        # Сигнал слишком тихий или нулевой
        return data


def normalize_audio_peak(data, target_peak=0.9, eps=1e-8, max_gain=100.0):
    """
    Нормализует аудиосигнал по пиковому значению.

    Args:
        data: numpy array с аудиоданными (float32)
        target_peak: целевое пиковое значение (0.0 - 1.0)
        eps: маленькое значение для избежания деления на ноль
        max_gain: максимальное линейное усиление (по умолчанию 100 = +40 dB)

    Returns:
        Нормализованный аудиосигнал
    """
    if len(data) == 0:
        return data

    # Находим максимальное абсолютное значение
    peak = np.max(np.abs(data))

    if peak > eps:
        # Вычисляем коэффициент усиления
        gain = target_peak / peak

        # Ограничиваем максимальное усиление
        gain = np.clip(gain, 0, max_gain)

        return data * gain
    else:
        # Сигнал слишком тихий или нулевой
        return data


# ==================== ЗАПИСЬ ЗВУКА ====================
class AudioRecorder:
    """Новая система записи аудио с ресемплингом в реальном времени."""

    def __init__(self, on_finish_callback, config=None):
        self.on_finish = on_finish_callback
        self.recording = False
        self.thread = None
        self.config = config or {}
        # Устанавливаем значения по умолчанию для конфига
        self.config.setdefault(
            "apply_window", False
        )  # Отключаем оконные функции по умолчанию
        self.config.setdefault("window_type", "hann")
        self.config.setdefault(
            "window_strength", 0.3
        )  # Менее агрессивное окно по умолчанию
        self.config.setdefault(
            "window_mic_only", True
        )  # Применять окно только к микрофону
        self.config.setdefault("blocksize", 16384)
        self.stop_event = threading.Event()

        # Ресурсы для очистки
        self._resources = {
            "pyaudio_instance": None,
            "sys_stream": None,
            "mic_stream": None,
            "mp3_file": None,
            "encoder": None,
            "debug_file": None,
        }

        # Буфер для мониторинга
        self.recent_audio_buffer = None
        self.recent_audio_lock = threading.Lock()

        # Частоты дискретизации
        self.base_sample_rate = None
        self.sys_sample_rate = None
        self.mic_sample_rate = None

        # Ресемплеры
        self.sys_resampler = None
        self.mic_resampler = None

        # Статистика
        self.stats = {
            "sys_samples": 0,
            "mic_samples": 0,
            "sys_resampled": 0,
            "mic_resampled": 0,
            "start_time": None,
            "end_time": None,
        }

    def _determine_sample_rates(self, p):
        """Определяет частоты дискретизации системного звука и микрофона."""
        try:
            # Получаем информацию о WASAPI
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)

            # Получаем устройства по умолчанию
            default_speakers = p.get_device_info_by_index(
                wasapi_info["defaultOutputDevice"]
            )
            default_mic = p.get_default_input_device_info()

            # Частота дискретизации системного звука (loopback)
            sys_sample_rate = float(default_speakers.get("defaultSampleRate", 44100.0))

            # Частота дискретизации микрофона
            mic_sample_rate = float(default_mic.get("defaultSampleRate", 44100.0))

            # Базовая частота - наименьшая из двух
            base_sample_rate = min(sys_sample_rate, mic_sample_rate)

            logging.info(f"Частота системного звука: {sys_sample_rate} Hz")
            logging.info(f"Частота микрофона: {mic_sample_rate} Hz")
            logging.info(f"Базовая частота записи: {base_sample_rate} Hz")

            return base_sample_rate, sys_sample_rate, mic_sample_rate

        except Exception as e:
            logging.error(f"Ошибка при определении частот дискретизации: {e}")
            # Возвращаем значения по умолчанию
            return 44100.0, 44100.0, 44100.0

    def _setup_resamplers(self):
        """Настраивает ресемплеры для потоков."""
        if self.sys_sample_rate != self.base_sample_rate:
            ratio = self.base_sample_rate / self.sys_sample_rate
            self.sys_resampler = samplerate.Resampler("sinc_fastest", channels=1)
            logging.info(
                f"Ресемплер системного звука: {self.sys_sample_rate} -> {self.base_sample_rate} (ratio: {ratio:.4f})"
            )
        else:
            self.sys_resampler = None
            logging.info("Ресемплинг системного звука не требуется")

        if self.mic_sample_rate != self.base_sample_rate:
            ratio = self.base_sample_rate / self.mic_sample_rate
            self.mic_resampler = samplerate.Resampler("sinc_fastest", channels=1)
            logging.info(
                f"Ресемплер микрофона: {self.mic_sample_rate} -> {self.base_sample_rate} (ratio: {ratio:.4f})"
            )
        else:
            self.mic_resampler = None
            logging.info("Ресемплинг микрофона не требуется")

    def _convert_to_mono(self, data, channels):
        """Преобразует многоканальные данные в моно."""
        if channels == 1:
            return data

        # Если данные имеют форму (samples, channels)
        if len(data.shape) == 2 and data.shape[1] == channels:
            return np.mean(data, axis=1)
        # Если данные имеют форму (channels, samples) - маловероятно для PyAudio
        elif len(data.shape) == 2 and data.shape[0] == channels:
            return np.mean(data, axis=0)
        else:
            # Неизвестный формат, возвращаем как есть
            logging.warning(
                f"Неизвестный формат данных: {data.shape}, channels={channels}"
            )
            return data

    def _resample_if_needed(self, data, resampler, original_rate, target_rate):
        """Выполняет ресемплинг данных если требуется."""
        if resampler is None or original_rate == target_rate:
            return data

        ratio = target_rate / original_rate
        try:
            resampled = resampler.process(data, ratio)
            return resampled
        except Exception as e:
            logging.error(f"Ошибка ресемплинга: {e}")
            return data

    def _find_loopback_device(self, p):
        """Находит loopback устройство для захвата системного звука."""
        # Пробуем использовать сохраненные настройки
        audio_output_device_id = self.config.get("audio_output_device_id", -1)
        audio_output_device = self.config.get("audio_output_device", "")

        # Сначала по индексу
        if audio_output_device_id >= 0:
            try:
                device = p.get_device_info_by_index(audio_output_device_id)
                for loopback in p.get_loopback_device_info_generator():
                    if loopback["index"] == audio_output_device_id:
                        logging.info(f"Найдено loopback по индексу: {loopback['name']}")
                        return loopback
            except Exception as e:
                logging.warning(
                    f"Ошибка при доступе к устройству {audio_output_device_id}: {e}"
                )

        # Затем по имени
        if audio_output_device:
            for loopback in p.get_loopback_device_info_generator():
                if audio_output_device.lower() in loopback["name"].lower():
                    logging.info(f"Найдено loopback по имени: {loopback['name']}")
                    return loopback

        # Пытаемся найти loopback устройство, соответствующее активному устройству вывода по умолчанию
        try:
            # Получаем информацию о WASAPI
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_output_index = wasapi_info["defaultOutputDevice"]
            default_output = p.get_device_info_by_index(default_output_index)
            default_output_name = default_output["name"]

            logging.info(
                f"Активное устройство вывода по умолчанию: {default_output_name}"
            )

            # Ищем loopback устройство с похожим именем
            for loopback in p.get_loopback_device_info_generator():
                loopback_name = loopback["name"]
                # Убираем "[Loopback]" из имени для сравнения
                clean_loopback_name = loopback_name.replace("[Loopback]", "").strip()
                if (
                    clean_loopback_name in default_output_name
                    or default_output_name in clean_loopback_name
                ):
                    logging.info(
                        f"Найдено соответствующее loopback устройство: {loopback_name}"
                    )
                    return loopback

                # Также проверяем частичное совпадение
                common_words = [
                    "speakers",
                    "headphones",
                    "audio",
                    "device",
                    "maxwell",
                    "game",
                    "chat",
                ]
                for word in common_words:
                    if (
                        word in default_output_name.lower()
                        and word in loopback_name.lower()
                    ):
                        logging.info(
                            f"Найдено loopback по ключевому слову '{word}': {loopback_name}"
                        )
                        return loopback
        except Exception as e:
            logging.warning(
                f"Не удалось найти loopback для активного устройства вывода: {e}"
            )

        # Собираем все доступные loopback устройства для выбора
        loopback_devices = list(p.get_loopback_device_info_generator())
        if loopback_devices:
            logging.info(f"Доступные loopback устройства ({len(loopback_devices)}):")
            for i, loopback in enumerate(loopback_devices):
                logging.info(f"  {i}: {loopback['name']}")

            # Пробуем найти устройство с "Game" или "Chat" в названии (скорее всего активное)
            for loopback in loopback_devices:
                loopback_name = loopback["name"].lower()
                if (
                    "game" in loopback_name
                    or "chat" in loopback_name
                    or "speakers" in loopback_name
                ):
                    logging.info(
                        f"Используем вероятно активное loopback устройство: {loopback['name']}"
                    )
                    return loopback

            # По умолчанию - первое доступное loopback устройство
            logging.info(
                f"Используется первое loopback устройство: {loopback_devices[0]['name']}"
            )
            return loopback_devices[0]

        raise RuntimeError(
            "Не найдено loopback устройство для захвата системного звука"
        )

    def _find_microphone_device(self, p):
        """Находит микрофонное устройство."""
        # Пробуем использовать сохраненные настройки
        audio_input_device_id = self.config.get("audio_input_device_id", -1)
        audio_input_device = self.config.get("audio_input_device", "")

        # Сначала по индексу
        if audio_input_device_id >= 0:
            try:
                device = p.get_device_info_by_index(audio_input_device_id)
                if device["maxInputChannels"] > 0:
                    logging.info(f"Найден микрофон по индексу: {device['name']}")
                    return device
            except Exception as e:
                logging.warning(
                    f"Ошибка при доступе к микрофону {audio_input_device_id}: {e}"
                )

        # Затем по имени
        if audio_input_device:
            for i in range(p.get_device_count()):
                device = p.get_device_info_by_index(i)
                if (
                    device["maxInputChannels"] > 0
                    and audio_input_device.lower() in device["name"].lower()
                ):
                    logging.info(f"Найден микрофон по имени: {device['name']}")
                    return device

        # Микрофон по умолчанию
        try:
            default_mic = p.get_default_input_device_info()
            logging.info(f"Используется микрофон по умолчанию: {default_mic['name']}")
            return default_mic
        except Exception as e:
            logging.error(f"Не удалось получить микрофон по умолчанию: {e}")
            raise RuntimeError("Не найдено микрофонное устройство")

    def start_recording(self, save_path, blocksize=None):
        """Начинает запись аудио."""
        self.recording = True
        self.stop_event.clear()
        self.mp3_path = save_path
        # Используем blocksize из конфига, если не передан явно
        if blocksize is None:
            self.blocksize = self.config.get("blocksize", 16384)
        else:
            self.blocksize = blocksize

        self.thread = threading.Thread(target=self._record)
        self.thread.daemon = True
        self.thread.start()

    def stop_recording(self):
        """Останавливает запись."""
        self.recording = False
        self.stop_event.set()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)

            if self.thread.is_alive():
                logging.warning("Поток записи не завершился, принудительная очистка...")
                self._cleanup_resources()

    def _cleanup_resources(self):
        """Очищает все ресурсы."""
        try:
            # Закрываем потоки с проверкой состояния
            for key in ["sys_stream", "mic_stream"]:
                stream = self._resources.get(key)
                if stream:
                    try:
                        # Пытаемся остановить поток, если он активен
                        if hasattr(stream, "is_active"):
                            if stream.is_active():
                                try:
                                    stream.stop_stream()
                                except Exception as stop_e:
                                    # Игнорируем ошибки остановки
                                    pass

                        # Закрываем поток
                        stream.close()
                    except Exception as e:
                        # Игнорируем ошибки "Stream not open" и подобные
                        error_str = str(e).lower()
                        if "not open" not in error_str and "closed" not in error_str:
                            logging.warning(f"Ошибка при закрытии потока {key}: {e}")
                    finally:
                        # Убираем ссылку
                        self._resources[key] = None

            # Закрываем файлы
            for key in ["mp3_file", "debug_file"]:
                file_obj = self._resources.get(key)
                if file_obj:
                    try:
                        file_obj.close()
                    except Exception as e:
                        logging.warning(f"Ошибка при закрытии файла {key}: {e}")
                    finally:
                        self._resources[key] = None

            # Завершаем PyAudio
            p = self._resources.get("pyaudio_instance")
            if p:
                try:
                    p.terminate()
                except Exception as e:
                    logging.warning(f"Ошибка при завершении PyAudio: {e}")
                finally:
                    self._resources["pyaudio_instance"] = None

        except Exception as e:
            logging.error(f"Ошибка при очистке ресурсов: {e}")

    def _record(self):
        """Основной метод записи."""
        logger = logging.getLogger(__name__)
        logger.info(f"Начало записи аудио в файл: {self.mp3_path}")

        self.stats["start_time"] = time.time()

        p = None
        sys_stream = None
        mic_stream = None
        mp3_file = None
        encoder = None
        recording_error = None

        try:
            # Инициализация PyAudio
            p = pyaudio.PyAudio()
            self._resources["pyaudio_instance"] = p

            # Определение частот дискретизации
            self.base_sample_rate, self.sys_sample_rate, self.mic_sample_rate = (
                self._determine_sample_rates(p)
            )

            # Настройка ресемплеров
            self._setup_resamplers()

            # Поиск устройств
            loopback_device = self._find_loopback_device(p)
            mic_device = self._find_microphone_device(p)

            # Открываем потоки
            sys_stream = p.open(
                format=pyaudio.paInt16,
                channels=loopback_device["maxInputChannels"],
                rate=int(self.sys_sample_rate),
                input=True,
                input_device_index=loopback_device["index"],
                frames_per_buffer=self.blocksize,
            )
            self._resources["sys_stream"] = sys_stream

            mic_stream = p.open(
                format=pyaudio.paInt16,
                channels=mic_device["maxInputChannels"],
                rate=int(self.mic_sample_rate),
                input=True,
                input_device_index=mic_device["index"],
                frames_per_buffer=self.blocksize,
            )
            self._resources["mic_stream"] = mic_stream

            # Создаем MP3 encoder
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(int(self.base_sample_rate))
            encoder.set_channels(2)  # Стерео: левый = системный звук, правый = микрофон
            encoder.set_quality(5)  # Качество кодирования (как в оригинальном коде)

            # Открываем файл для записи
            mp3_file = open(self.mp3_path, "wb")
            self._resources["mp3_file"] = mp3_file
            self._resources["encoder"] = encoder

            logger.info("Запись начата")

            # Основной цикл записи
            while self.recording and not self.stop_event.is_set():
                # Чтение данных из потоков
                sys_data = sys_stream.read(self.blocksize, exception_on_overflow=False)
                mic_data = mic_stream.read(self.blocksize, exception_on_overflow=False)

                # Преобразование в numpy массивы с правильной нормализацией
                sys_array = (
                    np.frombuffer(sys_data, dtype=np.int16).astype(np.float32) / 32767.0
                )
                mic_array = (
                    np.frombuffer(mic_data, dtype=np.int16).astype(np.float32) / 32767.0
                )

                # Проверка на пустые данные
                if len(sys_array) == 0 or len(mic_array) == 0:
                    continue

                # Преобразование в моно
                sys_channels = loopback_device["maxInputChannels"]
                mic_channels = mic_device["maxInputChannels"]

                # Правильное преобразование многоканальных данных в моно
                if sys_channels > 1:
                    sys_array_reshaped = sys_array.reshape(-1, sys_channels)
                    sys_mono = np.mean(sys_array_reshaped, axis=1)
                else:
                    sys_mono = sys_array

                if mic_channels > 1:
                    mic_array_reshaped = mic_array.reshape(-1, mic_channels)
                    mic_mono = np.mean(mic_array_reshaped, axis=1)
                else:
                    mic_mono = mic_array

                # Применяем оконную функцию для сглаживания краев (если включено в конфиге)
                if self.config.get("apply_window", False):
                    window_type = self.config.get("window_type", "hann")
                    window_strength = self.config.get("window_strength", 0.3)
                    window_mic_only = self.config.get("window_mic_only", True)

                    # Применяем окно только к микрофону или к обоим каналам
                    if window_mic_only:
                        # Только микрофон
                        mic_mono = apply_window(
                            mic_mono, window_type=window_type, strength=window_strength
                        )
                        # Системный звук без окна
                    else:
                        # Оба канала
                        sys_mono = apply_window(
                            sys_mono, window_type=window_type, strength=window_strength
                        )
                        mic_mono = apply_window(
                            mic_mono, window_type=window_type, strength=window_strength
                        )

                # Ресемплинг если требуется
                sys_resampled = self._resample_if_needed(
                    sys_mono,
                    self.sys_resampler,
                    self.sys_sample_rate,
                    self.base_sample_rate,
                )
                mic_resampled = self._resample_if_needed(
                    mic_mono,
                    self.mic_resampler,
                    self.mic_sample_rate,
                    self.base_sample_rate,
                )

                # Обновление статистики
                self.stats["sys_samples"] += len(sys_mono)
                self.stats["mic_samples"] += len(mic_mono)
                if self.sys_resampler:
                    self.stats["sys_resampled"] += len(sys_resampled)
                if self.mic_resampler:
                    self.stats["mic_resampled"] += len(mic_resampled)

                # Создание стерео данных (левый = системный, правый = микрофон)
                # Выравнивание длин массивов
                min_len = min(len(sys_resampled), len(mic_resampled))
                if min_len > 0:
                    # Создаем стерео массив
                    stereo_float = np.column_stack(
                        [sys_resampled[:min_len], mic_resampled[:min_len]]
                    ).flatten()

                    # Преобразуем float32 (-1.0 to 1.0) обратно в int16 для encoder
                    stereo_int16 = (stereo_float * 32767.0).astype(np.int16)

                    # Кодирование в MP3
                    mp3_chunk = encoder.encode(stereo_int16.tobytes())
                    if mp3_chunk:
                        mp3_file.write(mp3_chunk)

                # Обновление буфера для мониторинга
                with self.recent_audio_lock:
                    self.recent_audio_buffer = (
                        stereo_float[: min_len * 2].reshape(-1, 2)
                        if min_len > 0
                        else None
                    )

            # Завершение записи
            logger.info("Завершение записи...")

            # Финальный flush encoder
            try:
                final_chunk = encoder.flush()
                if final_chunk:
                    mp3_file.write(final_chunk)
            except Exception as e:
                logger.warning(f"Ошибка при flush encoder: {e}")

            self.stats["end_time"] = time.time()
            duration = self.stats["end_time"] - self.stats["start_time"]
            logger.info(f"Запись завершена. Длительность: {duration:.2f} сек")
            logger.info(
                f"Статистика: системных сэмплов: {self.stats['sys_samples']}, микрофонных: {self.stats['mic_samples']}"
            )

        except Exception as e:
            logger.error(f"Ошибка при записи: {e}")
            logger.error(traceback.format_exc())
            recording_error = str(e)
            # Устанавливаем end_time даже при ошибке
            if self.stats.get("end_time") is None:
                self.stats["end_time"] = time.time()
        finally:
            # Убедимся, что end_time установлен
            if self.stats.get("end_time") is None:
                self.stats["end_time"] = time.time()

            # Очистка ресурсов
            self._cleanup_resources()

            # Вызов callback
            if self.on_finish:
                try:
                    if recording_error:
                        # Была ошибка при записи
                        self.on_finish(False, None, recording_error)
                    else:
                        # Определяем успешность записи
                        success = (
                            self.stats.get("sys_samples", 0) > 0
                            or self.stats.get("mic_samples", 0) > 0
                        )
                        error_msg = None if success else "Запись не удалась: нет данных"
                        self.on_finish(success, self.mp3_path, error_msg)
                except Exception as e:
                    logger.error(f"Ошибка в callback: {e}")

    def get_recent_audio(self):
        """Возвращает последние аудиоданные для мониторинга."""
        with self.recent_audio_lock:
            return self.recent_audio_buffer

    def check_devices_available(self):
        """Проверяет доступность аудиоустройств."""
        p = None
        try:
            p = pyaudio.PyAudio()
            result = {
                "loopback_available": False,
                "microphone_available": False,
                "loopback_devices": [],
                "microphone_devices": [],
                "default_microphone": None,
                "default_speakers": None,
                "errors": [],
            }

            # Проверка WASAPI
            try:
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                result["wasapi_available"] = True
                result["default_speakers"] = p.get_device_info_by_index(
                    wasapi_info["defaultOutputDevice"]
                )
            except OSError as e:
                result["wasapi_available"] = False
                result["errors"].append(f"WASAPI недоступен: {e}")
                return result

            # Поиск loopback устройств
            try:
                loopback_devices = list(p.get_loopback_device_info_generator())
                result["loopback_devices"] = loopback_devices
                result["loopback_available"] = len(loopback_devices) > 0
            except Exception as e:
                result["errors"].append(f"Ошибка при поиске loopback устройств: {e}")

            # Поиск микрофонов
            mic_devices = []
            for i in range(p.get_device_count()):
                try:
                    device = p.get_device_info_by_index(i)
                    if device["maxInputChannels"] > 0:
                        mic_devices.append(device)
                except Exception as e:
                    result["errors"].append(f"Ошибка при получении устройства {i}: {e}")

            result["microphone_devices"] = mic_devices
            result["microphone_available"] = len(mic_devices) > 0

            # Микрофон по умолчанию
            try:
                result["default_microphone"] = p.get_default_input_device_info()
            except OSError as e:
                result["default_microphone"] = None
                result["errors"].append(f"Микрофон по умолчанию недоступен: {e}")

            return result
        except Exception as e:
            if p:
                p.terminate()
            raise
        finally:
            if p:
                p.terminate()

    def get_available_devices(self, force_refresh=False):
        """Возвращает структурированные данные об аудиоустройствах системы."""
        import time

        # Кэширование результатов
        if not hasattr(self, "_devices_cache"):
            self._devices_cache = None
            self._devices_cache_timestamp = 0

        cache_ttl = 30.0  # секунд

        if (
            not force_refresh
            and self._devices_cache is not None
            and (time.time() - self._devices_cache_timestamp) < cache_ttl
        ):
            return self._devices_cache

        p = None
        try:
            p = pyaudio.PyAudio()
            result = {
                "output_devices": [],
                "input_devices": [],
                "default_output_index": None,
                "default_input_index": None,
                "cache_timestamp": time.time(),
                "errors": [],
            }

            # Получаем устройства по умолчанию
            try:
                default_output = p.get_default_output_device_info()
                result["default_output_index"] = default_output["index"]
            except Exception as e:
                result["errors"].append(
                    f"Не удалось получить устройство вывода по умолчанию: {e}"
                )

            try:
                default_input = p.get_default_input_device_info()
                result["default_input_index"] = default_input["index"]
            except Exception as e:
                result["errors"].append(
                    f"Не удалось получить устройство ввода по умолчанию: {e}"
                )

            # Собираем информацию обо всех устройствах
            for i in range(p.get_device_count()):
                try:
                    device_info = p.get_device_info_by_index(i)

                    device_data = {
                        "name": device_info.get("name", f"Устройство {i}"),
                        "index": i,
                        "channels": int(
                            device_info.get("maxInputChannels", 0)
                            if device_info.get("maxInputChannels", 0) > 0
                            else device_info.get("maxOutputChannels", 0)
                        ),
                        "sample_rate": float(
                            device_info.get("defaultSampleRate", 44100.0)
                        ),
                        "default_sample_rate": float(
                            device_info.get("defaultSampleRate", 44100.0)
                        ),
                        "host_api": "WASAPI",
                        "is_default": (
                            i == result["default_output_index"]
                            or i == result["default_input_index"]
                        ),
                    }

                    # Определяем тип устройства
                    max_input = device_info.get("maxInputChannels", 0)
                    max_output = device_info.get("maxOutputChannels", 0)

                    if max_output > 0:
                        device_data["type"] = "output"
                        result["output_devices"].append(device_data)

                    if max_input > 0:
                        device_data["type"] = "input"
                        result["input_devices"].append(device_data)

                except Exception as e:
                    result["errors"].append(f"Устройство {i}: {e}")

            # Добавляем loopback устройства
            try:
                loopback_devices = list(p.get_loopback_device_info_generator())
                for lb_device in loopback_devices:
                    device_data = {
                        "name": lb_device.get("name", "Loopback") + " (Loopback)",
                        "index": lb_device.get("index", -1),
                        "channels": int(lb_device.get("maxInputChannels", 2)),
                        "sample_rate": float(
                            lb_device.get("defaultSampleRate", 44100.0)
                        ),
                        "default_sample_rate": float(
                            lb_device.get("defaultSampleRate", 44100.0)
                        ),
                        "host_api": "WASAPI Loopback",
                        "is_default": False,
                        "type": "loopback",
                        "is_loopback": True,
                    }
                    result["output_devices"].append(device_data)
            except Exception as e:
                result["errors"].append(f"Loopback устройства: {e}")

            # Сохраняем в кэш
            self._devices_cache = result
            self._devices_cache_timestamp = time.time()

            return result

        except Exception as e:
            if p:
                p.terminate()
            raise
        finally:
            if p:
                p.terminate()
