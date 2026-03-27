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
    def __init__(self, on_finish_callback, config=None):
        self.on_finish = on_finish_callback
        self.recording = False
        self.thread = None
        self.config = config or {}
        self.stop_event = threading.Event()
        self._force_cleanup_resources = {}
        # Буфер для мониторинга VU-метра
        self.recent_audio_buffer = (
            None  # последние аудиоданные (форма (samples, channels))
        )
        self.recent_audio_lock = threading.Lock()

    def _get_whisper_json_path(self, audio_path):
        """Возвращает ожидаемый путь к JSON-файлу Whisper для данного аудио."""
        folder = self.config.get("whisper_output_folder", "")
        if not folder:
            folder = os.path.dirname(audio_path)
        base_name = Path(audio_path).stem
        return os.path.join(folder, f"{base_name}.json")

    def get_recent_audio(self):
        """
        Возвращает последние аудиоданные для мониторинга VU-метром.
        Возвращает numpy array формы (samples, channels) или None, если данных нет.
        """
        with self.recent_audio_lock:
            return self.recent_audio_buffer

    def check_devices_available(self):
        """
        Проверяет доступность аудиоустройств (loopback и микрофона).
        Возвращает словарь с информацией о доступных устройствах.
        """
        import pyaudiowpatch as pyaudio

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
            loopback_devices = list(p.get_loopback_device_info_generator())
            result["loopback_devices"] = loopback_devices
            result["loopback_available"] = len(loopback_devices) > 0

            # Поиск микрофонов
            mic_devices = []
            for i in range(p.get_device_count()):
                device = p.get_device_info_by_index(i)
                if device["maxInputChannels"] > 0:
                    mic_devices.append(device)
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
        """
        Возвращает структурированные данные об аудиоустройствах системы.

        Args:
            force_refresh (bool): Если True, игнорирует кэш и обновляет список устройств.

        Returns:
            dict: Словарь с отдельными списками для выходных (loopback) и входных (микрофон) устройств.
                Структура:
                {
                    "output_devices": [
                        {
                            "name": str,
                            "index": int,
                            "channels": int,
                            "sample_rate": float,
                            "default_sample_rate": float,
                            "host_api": str,
                            "is_default": bool
                        },
                        ...
                    ],
                    "input_devices": [
                        {
                            "name": str,
                            "index": int,
                            "channels": int,
                            "sample_rate": float,
                            "default_sample_rate": float,
                            "host_api": str,
                            "is_default": bool
                        },
                        ...
                    ],
                    "default_output_index": int,
                    "default_input_index": int,
                    "cache_timestamp": float,
                    "errors": list[str]
                }
        """
        import pyaudiowpatch as pyaudio
        import time
        import logging

        logger = logging.getLogger(__name__)

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
            # logger.debug("Используется кэшированный список устройств")  # сокращение логов
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

            # Получаем информацию о хостах
            host_apis = {}
            for i in range(p.get_host_api_count()):
                try:
                    host_info = p.get_host_api_info_by_index(i)
                    host_apis[host_info["index"]] = host_info["name"]
                except Exception as e:
                    logger.warning(f"Не удалось получить информацию о хосте {i}: {e}")

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
                        "host_api": host_apis.get(
                            device_info.get("hostApi", -1), "Unknown"
                        ),
                        "is_default": (
                            i == result["default_output_index"]
                            or i == result["default_input_index"]
                        ),
                    }

                    # Определяем тип устройства
                    max_input = device_info.get("maxInputChannels", 0)
                    max_output = device_info.get("maxOutputChannels", 0)

                    if max_output > 0:
                        # Это устройство вывода (может быть loopback)
                        device_data["type"] = "output"
                        result["output_devices"].append(device_data)

                    if max_input > 0:
                        # Это устройство ввода (микрофон)
                        device_data["type"] = "input"
                        result["input_devices"].append(device_data)

                except Exception as e:
                    logger.error(
                        f"Ошибка при получении информации об устройстве {i}: {e}"
                    )
                    result["errors"].append(f"Устройство {i}: {e}")

            # Добавляем loopback устройства (специальные устройства записи вывода)
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
                logger.warning(f"Не удалось получить loopback устройства: {e}")
                result["errors"].append(f"Loopback устройства: {e}")

            # Сохраняем в кэш
            self._devices_cache = result
            self._devices_cache_timestamp = time.time()

            logger.info(
                f"Получено {len(result['output_devices'])} выходных и {len(result['input_devices'])} входных устройств"
            )
            return result

        except Exception as e:
            logger.error(f"Критическая ошибка при получении списка устройств: {e}")
            if p:
                p.terminate()
            raise
        finally:
            if p:
                p.terminate()

    def start_recording(self, save_path, blocksize=1024):
        self.recording = True
        self.stop_event.clear()
        self.mp3_path = save_path
        self.blocksize = blocksize
        self.thread = threading.Thread(target=self._record)
        self.thread.daemon = True
        self.thread.start()

    def stop_recording(self):
        """Останавливает запись с гарантированным освобождением ресурсов.

        Использует упрощенную многоуровневую стратегию остановки:
        1. Установка флагов остановки
        2. Ожидание корректного завершения с таймаутом
        3. Принудительное прерывание потоков PyAudio при зависании
        4. Гарантированная очистка ресурсов
        """
        import logging

        logger = logging.getLogger(__name__)

        self.recording = False
        self.stop_event.set()

        # Сохраняем ссылки на ресурсы для принудительной очистки
        # (будут установлены в _record)
        self._force_cleanup_resources = getattr(self, "_force_cleanup_resources", {})

        # Логирование состояния потоков перед остановкой
        resources = self._force_cleanup_resources
        if "sys_stream" in resources and resources["sys_stream"]:
            stream = resources["sys_stream"]
            is_active = hasattr(stream, "is_active") and stream.is_active()
            # logger.debug(f"Состояние системного потока перед остановкой: активен={is_active}")  # сокращение логов
        if "mic_stream" in resources and resources["mic_stream"]:
            stream = resources["mic_stream"]
            is_active = hasattr(stream, "is_active") and stream.is_active()
            # logger.debug(f"Состояние микрофонного потока перед остановкой: активен={is_active}")  # сокращение логов

        # Ожидание корректного завершения с прогрессивными таймаутами
        if self.thread and self.thread.is_alive():
            logger.info("Ожидание завершения потока записи...")

            # Первая попытка: короткий таймаут для быстрого завершения
            self.thread.join(timeout=1.5)

            if self.thread.is_alive():
                logger.warning(
                    "Поток не завершился за 1.5 секунды, попытка принудительной остановки потоков PyAudio..."
                )

                # Вторая попытка: принудительная остановка потоков PyAudio
                self._abort_pyaudio_streams()

                # Даем время на реакцию
                self.thread.join(timeout=1.0)

                if self.thread.is_alive():
                    logger.error(
                        "Поток все еще жив после принудительной остановки, применение агрессивной очистки..."
                    )

                    # Третья попытка: агрессивная очистка ресурсов
                    self._guaranteed_cleanup()

                    # Финальное ожидание
                    self.thread.join(timeout=0.5)

                    if self.thread.is_alive():
                        logger.critical(
                            "Поток записи не отвечает даже после агрессивной очистки. Возможно зависание в системном вызове."
                        )
                        # На этом этапе мы не можем сделать больше, но гарантируем,
                        # что ресурсы были освобождены насколько это возможно
                    else:
                        logger.info("Поток завершен после агрессивной очистки.")
                else:
                    logger.info(
                        "Поток завершен после принудительной остановки потоков PyAudio."
                    )
            else:
                logger.info("Поток записи корректно завершен.")
        else:
            logger.info("Поток записи не активен.")

    def _flush_with_timeout(self, encoder, timeout=2.0):
        """
        Выполняет encoder.flush() с ограничением по времени и механизмом принудительного прерывания.

        Использует прогрессивную стратегию:
        1. Попытка стандартного flush с таймаутом
        2. При зависании - попытка прервать через отдельный поток
        3. Принудительное завершение с восстановлением управления

        Args:
            encoder: экземпляр lameenc.Encoder
            timeout: максимальное время ожидания в секундах

        Returns:
            bytes или None: данные MP3 или None в случае таймаута/ошибки
        """
        import threading
        import time
        import logging

        logger = logging.getLogger(__name__)

        result = []
        error = []
        flush_completed = threading.Event()

        def worker():
            try:
                # logger.debug("Начало encoder.flush()...")  # сокращение логов
                chunk = encoder.flush()
                result.append(chunk)
                # logger.debug("encoder.flush() успешно завершен")  # сокращение логов
            except Exception as e:
                error.append(e)
                logger.error(f"Ошибка в encoder.flush(): {e}")
            finally:
                flush_completed.set()

        # Запускаем поток с flush
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

        # Ожидаем завершения с прогрессивными таймаутами
        wait_time = timeout
        start_time = time.time()

        while wait_time > 0 and not flush_completed.is_set():
            # Проверяем каждые 0.1 секунды для более отзывчивого прерывания
            check_interval = 0.1
            flush_completed.wait(check_interval)
            elapsed = time.time() - start_time
            wait_time = timeout - elapsed

            # Проверяем stop_event для возможности прервать операцию
            if hasattr(self, "stop_event") and self.stop_event.is_set():
                logger.warning(
                    "Обнаружен stop_event во время flush, попытка прервать операцию..."
                )
                # Прерываем ожидание, но даем flush шанс завершиться
                break

        if thread.is_alive():
            # Поток все еще работает - зависание
            logger.warning(
                f"encoder.flush() завис после {elapsed:.1f} секунд, применение стратегии восстановления..."
            )

            # Стратегия 1: Попытка мягкого прерывания через stop_event
            if hasattr(self, "stop_event"):
                self.stop_event.set()
                # Даем дополнительное время для реакции
                thread.join(timeout=0.5)

            if thread.is_alive():
                # Стратегия 2: Принудительное завершение потока (опасно, но необходимо)
                logger.error("Принудительное завершение потока flush из-за зависания")
                # В Python нет безопасного способа убить поток, но мы можем
                # пометить операцию как неудачную и продолжить
                # Записываем предупреждение в лог
                return None
            else:
                logger.info("Поток flush завершился после мягкого прерывания")
                return result[0] if result else None
        elif error:
            # Произошла ошибка при flush
            logger.error(f"Ошибка при encoder.flush(): {error[0]}")
            return None
        else:
            # Успешное завершение
            # logger.debug(f"encoder.flush() завершен за {elapsed:.1f} секунд")  # сокращение логов
            return result[0] if result else None

    def _safe_stream_read(self, stream, num_frames, timeout_ms=1000):
        """
        Безопасное чтение из потока PyAudio с таймаутом и проверкой stop_event.

        Args:
            stream: поток PyAudio (может быть None)
            num_frames: количество фреймов для чтения
            timeout_ms: таймаут в миллисекундах

        Returns:
            tuple: (data, overflow) или (None, False) в случае таймаута/ошибки
        """
        import time
        import logging

        logger = logging.getLogger(__name__)

        # Если поток None (микрофон недоступен), возвращаем None
        if stream is None:
            logger.debug("Поток None, пропускаем чтение")
            return None, False

        # Проверяем, активен ли поток перед чтением
        if hasattr(stream, "is_active") and not stream.is_active():
            logger.warning("Попытка чтения из неактивного потока")
            return None, False

        # Проверяем stop_event перед чтением
        if hasattr(self, "stop_event") and self.stop_event.is_set():
            logger.debug("stop_event установлен, пропускаем чтение")
            return None, False

        start_time = time.time()
        logger.debug(
            f"Начало чтения из потока, num_frames={num_frames}, timeout={timeout_ms}мс"
        )
        # Запись в debug_file удалена для сокращения логов

        # Пытаемся прочитать с таймаутом и проверкой доступных данных
        try:
            # Проверяем, есть ли метод get_stream_read_available
            available = 0
            if hasattr(stream, "get_stream_read_available"):
                available = stream.get_stream_read_available()
                logger.debug(f"Доступно фреймов для чтения: {available}")
                # Запись в debug_file удалена для сокращения логов

            # Если доступных данных недостаточно, ждем с проверкой stop_event
            wait_start = time.time()
            while available < num_frames:
                # Проверяем таймаут
                if (time.time() - start_time) * 1000 > timeout_ms:
                    logger.warning(
                        f"Таймаут ожидания данных ({timeout_ms} мс), доступно только {available} фреймов"
                    )
                    return None, False
                # Проверяем stop_event
                if hasattr(self, "stop_event") and self.stop_event.is_set():
                    logger.debug("stop_event установлен во время ожидания данных")
                    return None, False
                # Короткая пауза
                time.sleep(0.001)  # 1 мс
                if hasattr(stream, "get_stream_read_available"):
                    available = stream.get_stream_read_available()
                else:
                    # Если метод недоступен, прерываем цикл
                    break

            # Чтение данных
            result = stream.read(
                num_frames,
                exception_on_overflow=False,
            )

            elapsed = time.time() - start_time
            logger.debug(
                f"Чтение завершено за {elapsed*1000:.0f} мс, размер данных: {len(result) if result else 0} байт"
            )
            return result, False

        except OSError as e:
            # Специфические ошибки PyAudio (устройство отключено, проблемы с драйвером)
            logger.error(f"Ошибка ввода-вывода PyAudio при чтении из потока: {e}")
            return None, False
        except IOError as e:
            # Устаревшее исключение, но оставляем для совместимости
            logger.error(f"Ошибка ввода-вывода при чтении из потока: {e}")
            return None, False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при чтении из потока: {e}")
            return None, False

    def _abort_pyaudio_streams(self):
        """
        Принудительное прерывание потоков PyAudio.

        Использует abort_stream() если доступно, иначе stop_stream().
        Гарантирует, что потоки будут остановлены даже при зависании.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Получаем ссылки на ресурсы из _record
        resources = getattr(self, "_force_cleanup_resources", {})

        logger.info("Принудительная остановка потоков PyAudio...")

        # Останавливаем системный поток
        if "sys_stream" in resources and resources["sys_stream"]:
            try:
                stream = resources["sys_stream"]
                # logger.debug(f"Принудительная остановка sys_stream, тип: {type(stream)}")  # сокращение логов
                # Безопасная проверка активности потока
                is_active = False
                if hasattr(stream, "is_active"):
                    try:
                        is_active = stream.is_active()
                        # logger.debug(f"sys_stream is_active() = {is_active}")  # сокращение логов
                    except OSError as e:
                        if "Stream not open" in str(e):
                            # logger.debug("sys_stream уже закрыт (Stream not open), считаем неактивным")  # сокращение логов
                            is_active = False
                        else:
                            raise

                if is_active:
                    if hasattr(stream, "abort_stream"):
                        stream.abort_stream()
                        # logger.debug("Системный поток прерван через abort_stream()")  # сокращение логов
                    else:
                        stream.stop_stream()
                        # logger.debug("Системный поток остановлен через stop_stream()")  # сокращение логов
                else:
                    # logger.debug("Системный поток уже не активен, пропускаем остановку")  # сокращение логов
                    pass
            except Exception as e:
                logger.error(f"Ошибка при остановке системного потока: {e}")

        # Останавливаем микрофонный поток
        if "mic_stream" in resources and resources["mic_stream"]:
            try:
                stream = resources["mic_stream"]
                # logger.debug(f"Принудительная остановка mic_stream, тип: {type(stream)}")  # сокращение логов
                # Безопасная проверка активности потока
                is_active = False
                if hasattr(stream, "is_active"):
                    try:
                        is_active = stream.is_active()
                        # logger.debug(f"mic_stream is_active() = {is_active}")  # сокращение логов
                    except OSError as e:
                        if "Stream not open" in str(e):
                            # logger.debug("mic_stream уже закрыт (Stream not open), считаем неактивным")  # сокращение логов
                            is_active = False
                        else:
                            raise

                if is_active:
                    if hasattr(stream, "abort_stream"):
                        stream.abort_stream()
                        # logger.debug("Микрофонный поток прерван через abort_stream()")  # сокращение логов
                    else:
                        stream.stop_stream()
                        # logger.debug("Микрофонный поток остановлен через stop_stream()")  # сокращение логов
                else:
                    # logger.debug("Микрофонный поток уже не активен, пропускаем остановку")  # сокращение логов
                    pass
            except Exception as e:
                logger.error(f"Ошибка при остановке микрофонного потока: {e}")

        # Закрываем PyAudio instance
        if "pyaudio_instance" in resources and resources["pyaudio_instance"]:
            try:
                p = resources["pyaudio_instance"]
                p.terminate()
                # logger.debug("Экземпляр PyAudio завершен")  # сокращение логов
            except Exception as e:
                logger.error(f"Ошибка при завершении PyAudio: {e}")

    def _guaranteed_cleanup(self):
        """
        Гарантированное освобождение всех ресурсов даже при ошибках.

        Выполняет:
        1. Закрытие всех потоков PyAudio
        2. Закрытие файлов
        3. Освобождение других ресурсов
        4. Логирование состояния
        """
        import logging
        import traceback

        logger = logging.getLogger(__name__)
        logger.info("Выполнение гарантированной очистки ресурсов...")

        # Получаем ссылки на ресурсы из _record
        resources = getattr(self, "_force_cleanup_resources", {})

        # 1. Закрытие файлов
        file_resources = ["mp3_file", "raw_mic_file", "debug_file"]
        for resource_name in file_resources:
            if resource_name in resources and resources[resource_name]:
                try:
                    file_obj = resources[resource_name]
                    if hasattr(file_obj, "close") and not file_obj.closed:
                        file_obj.close()
                        logger.debug(f"Файл {resource_name} закрыт")
                except Exception as e:
                    logger.error(f"Ошибка при закрытии файла {resource_name}: {e}")
                    traceback.print_exc()

        # 2. Остановка и закрытие потоков PyAudio
        stream_resources = ["sys_stream", "mic_stream"]
        for stream_name in stream_resources:
            if stream_name in resources and resources[stream_name]:
                try:
                    stream = resources[stream_name]
                    logger.debug(f"Обработка потока {stream_name}, тип: {type(stream)}")

                    # Безопасная проверка активности потока с обработкой OSError
                    is_active = False
                    if hasattr(stream, "is_active"):
                        try:
                            is_active = stream.is_active()
                            logger.debug(
                                f"Поток {stream_name} is_active() = {is_active}"
                            )
                        except OSError as e:
                            if "Stream not open" in str(e):
                                logger.debug(
                                    f"Поток {stream_name} уже закрыт (Stream not open), считаем неактивным"
                                )
                                is_active = False
                            else:
                                raise

                    # Останавливаем поток только если активен
                    if is_active:
                        if hasattr(stream, "abort_stream"):
                            stream.abort_stream()
                            logger.debug(
                                f"Поток {stream_name} прерван через abort_stream()"
                            )
                        elif hasattr(stream, "stop_stream"):
                            stream.stop_stream()
                            logger.debug(
                                f"Поток {stream_name} остановлен через stop_stream()"
                            )
                    else:
                        logger.debug(
                            f"Поток {stream_name} уже не активен, пропускаем остановку"
                        )

                    # Закрываем поток
                    if hasattr(stream, "close"):
                        stream.close()
                        logger.debug(f"Поток {stream_name} закрыт")

                    logger.debug(f"Поток {stream_name} обработка завершена")
                except Exception as e:
                    logger.error(f"Ошибка при остановке потока {stream_name}: {e}")
                    traceback.print_exc()

        # 3. Завершение экземпляра PyAudio
        if "pyaudio_instance" in resources and resources["pyaudio_instance"]:
            try:
                p = resources["pyaudio_instance"]
                if hasattr(p, "terminate"):
                    p.terminate()
                logger.debug("Экземпляр PyAudio завершен")
            except Exception as e:
                logger.error(f"Ошибка при завершении PyAudio: {e}")
                traceback.print_exc()

        # 4. Освобождение других ресурсов
        if "encoder" in resources and resources["encoder"]:
            try:
                # LAME encoder не требует явного закрытия
                pass
            except Exception as e:
                logger.error(f"Ошибка при освобождении encoder: {e}")

        logger.info("Гарантированная очистка ресурсов завершена")

    def _record(self):
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Начало записи аудио в файл: {self.mp3_path}")

        log_dir = os.path.join(os.path.dirname(self.mp3_path), "log")
        os.makedirs(log_dir, exist_ok=True)
        debug_filename = os.path.join(
            log_dir, f"debug_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        debug_file = open(debug_filename, "w", encoding="utf-8")
        debug_file.write("=== НАЧАЛО ЗАПИСИ (ПОТОКОВЫЙ РЕСЕМПЛИНГ) ===\n")
        self._force_cleanup_resources["debug_file"] = debug_file
        self.debug_file = debug_file  # для доступа из _safe_stream_read

        # Инициализация словаря для гарантированной очистки ресурсов
        self._force_cleanup_resources = (
            self._force_cleanup_resources
        )  # уже инициализирован

        p = None
        sys_stream = None
        mic_stream = None
        mp3_file = None
        encoder = None
        raw_mic_enabled = self.config.get("raw_microphone_recording", False)
        raw_mic_file = None
        try:
            debug_file.write("1. Инициализация PyAudio...\n")
            p = pyaudio.PyAudio()
            self._force_cleanup_resources["pyaudio_instance"] = p

            # --- Определяем устройства ---
            debug_file.write("2. Получение WASAPI хоста...\n")
            try:
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            except OSError as e:
                raise RuntimeError(f"WASAPI не доступен: {e}")

            default_speakers = p.get_device_info_by_index(
                wasapi_info["defaultOutputDevice"]
            )
            debug_file.write(f"   Динамики: {default_speakers['name']}\n")

            # Поиск loopback с поддержкой сохраненных настроек аудио
            loopback_device = None
            audio_output_device_id = self.config.get("audio_output_device_id", -1)
            audio_output_device = self.config.get("audio_output_device", "")

            # Сначала попробуем использовать сохраненный индекс, если он указан и является loopback
            if audio_output_device_id >= 0:
                debug_file.write(
                    f"   Проверка сохраненного выходного устройства по индексу: {audio_output_device_id}\n"
                )
                try:
                    device = p.get_device_info_by_index(audio_output_device_id)
                    # Проверяем, является ли устройство loopback (имеет входные каналы и имя содержит loopback?)
                    # Просто проверим, есть ли устройство в списке loopback
                    for loopback in p.get_loopback_device_info_generator():
                        if loopback["index"] == audio_output_device_id:
                            loopback_device = loopback
                            debug_file.write(
                                f"   Найдено loopback по сохраненному индексу: {loopback['name']}\n"
                            )
                            break
                    if loopback_device is None:
                        debug_file.write(
                            f"   Устройство с индексом {audio_output_device_id} не является loopback устройством\n"
                        )
                except OSError as e:
                    debug_file.write(
                        f"   Ошибка при доступе к устройству с индексом {audio_output_device_id}: {e}\n"
                    )

            # Если не нашли по индексу, ищем по имени
            if loopback_device is None and audio_output_device:
                debug_file.write(
                    f"   Поиск loopback по имени: '{audio_output_device}'\n"
                )
                for loopback in p.get_loopback_device_info_generator():
                    if audio_output_device.lower() in loopback["name"].lower():
                        loopback_device = loopback
                        debug_file.write(
                            f"   Найдено loopback по имени: {loopback['name']}\n"
                        )
                        break

            # Если всё ещё не найдено, используем старую логику (по умолчанию)
            if loopback_device is None:
                debug_file.write(
                    "   Поиск loopback по умолчанию (по имени динамиков)\n"
                )
                for loopback in p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        loopback_device = loopback
                        debug_file.write(f"   Найдено loopback: {loopback['name']}\n")
                        break

            if loopback_device is None:
                raise RuntimeError("Не найдено устройство захвата системного звука.")

            # Микрофон - улучшенный выбор с поддержкой новых настроек аудио
            mic_info = None
            self._mic_unavailable = False

            # Получаем настройки из конфигурации
            audio_input_device_id = self.config.get("audio_input_device_id", -1)
            audio_input_device = self.config.get("audio_input_device", "")
            mic_device_name = self.config.get(
                "microphone_device_name", ""
            )  # для обратной совместимости

            # Логируем только выбранные устройства (сокращение объема логов)

            # Шаг 1: Попытка использовать сохраненный индекс устройства
            if audio_input_device_id >= 0:
                debug_file.write(
                    f"   Поиск микрофона по индексу: {audio_input_device_id}\n"
                )
                try:
                    device = p.get_device_info_by_index(audio_input_device_id)
                    if device["maxInputChannels"] > 0:
                        mic_info = device
                        debug_file.write(
                            f"   Найден микрофон по индексу: {device['name']}\n"
                        )
                    else:
                        debug_file.write(
                            f"   Устройство с индексом {audio_input_device_id} не является микрофоном (нет входных каналов)\n"
                        )
                except OSError as e:
                    debug_file.write(
                        f"   Ошибка при доступе к устройству с индексом {audio_input_device_id}: {e}\n"
                    )

            # Шаг 2: Если индекс не сработал, ищем по имени
            if mic_info is None and audio_input_device:
                debug_file.write(
                    f"   Поиск микрофона по имени: '{audio_input_device}'\n"
                )
                for i in range(p.get_device_count()):
                    device = p.get_device_info_by_index(i)
                    if (
                        device["maxInputChannels"] > 0
                        and audio_input_device.lower() in device["name"].lower()
                    ):
                        mic_info = device
                        debug_file.write(
                            f"   Найден микрофон по имени: {device['name']}\n"
                        )
                        break

            # Шаг 3: Обратная совместимость с старым ключом microphone_device_name
            if mic_info is None and mic_device_name:
                debug_file.write(
                    f"   Поиск микрофона по устаревшему имени: '{mic_device_name}'\n"
                )
                for i in range(p.get_device_count()):
                    device = p.get_device_info_by_index(i)
                    if (
                        device["maxInputChannels"] > 0
                        and mic_device_name.lower() in device["name"].lower()
                    ):
                        mic_info = device
                        debug_file.write(f"   Найден микрофон: {device['name']}\n")
                        break

            # Шаг 4: Если не найден по имени или имя не указано, используем устройство по умолчанию
            if mic_info is None:
                debug_file.write("   Используется микрофон по умолчанию\n")
                try:
                    mic_info = p.get_default_input_device_info()
                except OSError as e:
                    debug_file.write(
                        f"   ОШИБКА: микрофон по умолчанию недоступен: {e}\n"
                    )
                    # Создаем заглушку для микрофона с параметрами системного устройства
                    # чтобы запись могла продолжиться (только системный звук)
                    debug_file.write("   СОЗДАНИЕ ЗАГЛУШКИ ДЛЯ МИКРОФОНА\n")
                    # Используем параметры системного устройства как заглушку
                    mic_info = {
                        "name": "VIRTUAL_MICROPHONE (заглушка)",
                        "index": -1,
                        "defaultSampleRate": loopback_device["defaultSampleRate"],
                        "maxInputChannels": 1,
                        "hostApi": 0,
                    }
                    # Устанавливаем флаг, что микрофон недоступен
                    self._mic_unavailable = True
                else:
                    self._mic_unavailable = False
            else:
                self._mic_unavailable = False

            debug_file.write(
                f"   Выбранный микрофон: {mic_info['name']} (индекс: {mic_info['index']})\n"
            )

            # Параметры устройств
            fs_sys = int(loopback_device["defaultSampleRate"])
            fs_mic = int(mic_info["defaultSampleRate"])
            channels_sys = loopback_device["maxInputChannels"]
            channels_mic = mic_info["maxInputChannels"]
            debug_file.write(f"   Системный звук: {fs_sys} Гц, {channels_sys} канала\n")
            debug_file.write(f"   Микрофон: {fs_mic} Гц, {channels_mic} канала\n")

            # Валидация параметров микрофона
            mic_parameters_valid = True
            if fs_mic <= 0:
                debug_file.write(
                    f"   ПРЕДУПРЕЖДЕНИЕ: некорректная частота дискретизации микрофона ({fs_mic} Гц)\n"
                )
                mic_parameters_valid = False
            if channels_mic <= 0:
                debug_file.write(
                    f"   ПРЕДУПРЕЖДЕНИЕ: некорректное количество каналов микрофона ({channels_mic})\n"
                )
                mic_parameters_valid = False
            if not mic_parameters_valid:
                debug_file.write(
                    "   Параметры микрофона некорректны, будет использоваться заглушка (нулевые данные).\n"
                )
                # Устанавливаем флаг, что микрофон недоступен
                self._mic_unavailable = True
                # Принудительно устанавливаем индекс -1, чтобы поток не открывался
                mic_info["index"] = -1

            # Целевая частота – максимальная из двух
            target_fs = max(fs_sys, fs_mic)
            debug_file.write(f"   Целевая частота: {target_fs} Гц\n")

            # --- Открываем потоки ---
            debug_file.write("3. Открытие потоков ввода...\n")
            sys_stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels_sys,
                rate=fs_sys,
                frames_per_buffer=0,  # используем оптимальный размер буфера
                input=True,
                input_device_index=loopback_device["index"],
            )

            # Проверяем индекс микрофона: если -1, микрофон недоступен, используем заглушку
            mic_stream = None
            if mic_info["index"] >= 0:
                mic_stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=channels_mic,
                    rate=fs_mic,
                    frames_per_buffer=0,  # используем оптимальный размер буфера
                    input=True,
                    input_device_index=mic_info["index"],
                )
                debug_file.write("   Микрофонный поток открыт.\n")
            else:
                debug_file.write(
                    "   Микрофон недоступен (индекс -1), будет использоваться заглушка (нулевые данные).\n"
                )
                # Устанавливаем флаг, что микрофон недоступен
                self._mic_unavailable = True

            debug_file.write("   Потоки открыты.\n")

            # Проверяем активность потоков
            if hasattr(sys_stream, "is_active"):
                debug_file.write(
                    f"   Системный поток активен: {sys_stream.is_active()}\n"
                )
            if mic_stream and hasattr(mic_stream, "is_active"):
                debug_file.write(
                    f"   Микрофонный поток активен: {mic_stream.is_active()}\n"
                )

            # Сохраняем ссылки на потоки для гарантированной очистки
            self._force_cleanup_resources["sys_stream"] = sys_stream
            self._force_cleanup_resources["mic_stream"] = mic_stream

            # --- Инициализация ресемплеров ---
            sys_ratio = target_fs / fs_sys if fs_sys != target_fs else 1.0
            mic_ratio = target_fs / fs_mic if fs_mic != target_fs else 1.0
            sys_resampler = (
                create_resampler("sinc_best", channels=1)
                if fs_sys != target_fs
                else None
            )
            mic_resampler = (
                create_resampler("sinc_best", channels=1)
                if fs_mic != target_fs
                else None
            )

            # --- Инициализация MP3-энкодера ---
            debug_file.write("4. Инициализация LAME-энкодера...\n")
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(target_fs)
            encoder.set_channels(1)
            encoder.set_quality(5)
            self._force_cleanup_resources["encoder"] = encoder

            mp3_file = open(self.mp3_path, "wb")
            debug_file.write(f"   MP3-файл открыт: {self.mp3_path}\n")
            self._force_cleanup_resources["mp3_file"] = mp3_file

            # Открытие файла для записи сырых данных микрофона (если включено)
            # raw_mic_enabled уже инициализирован перед try-блоком
            if raw_mic_enabled:
                raw_mic_path = os.path.join(
                    log_dir, f"raw_mic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
                )
                raw_mic_file = open(raw_mic_path, "wb")
                debug_file.write(f"   Файл сырых данных микрофона: {raw_mic_path}\n")
                self._force_cleanup_resources["raw_mic_file"] = raw_mic_file

            # --- Цикл записи ---
            debug_file.write("5. Начало цикла записи...\n")
            block_counter = 0
            start_time = time.time()
            last_log_time = start_time

            while self.recording and not self.stop_event.is_set():
                try:
                    # Логирование входа в итерацию (первые 5 итераций)
                    if block_counter < 5:
                        debug_file.write(
                            f"   Итерация {block_counter}, recording={self.recording}, stop_event={self.stop_event.is_set()}\n"
                        )
                        debug_file.flush()

                    # Безопасное чтение с проверкой stop_event между операциями
                    sys_result, sys_overflow = self._safe_stream_read(
                        sys_stream, self.blocksize, timeout_ms=500
                    )

                    # Проверяем stop_event между чтениями
                    if self.stop_event.is_set():
                        debug_file.write(
                            "   Обнаружен stop_event после чтения системного звука, прерывание цикла.\n"
                        )
                        break

                    # Чтение микрофона: если поток отсутствует (микрофон недоступен), используем нулевые данные
                    if mic_stream is not None:
                        mic_result, mic_overflow = self._safe_stream_read(
                            mic_stream, self.blocksize, timeout_ms=500
                        )
                    else:
                        # Генерируем нулевые данные для микрофона
                        mic_result = bytes(
                            self.blocksize * channels_mic * 4
                        )  # float32: 4 байта на сэмпл
                        mic_overflow = False
                        # Логируем только первые несколько блоков, чтобы не засорять лог
                        if block_counter < 5:
                            debug_file.write(
                                "   Используются нулевые данные для микрофона (микрофон недоступен).\n"
                            )

                    # Проверяем результаты чтения
                    if sys_result is None:
                        debug_file.write(
                            "   Таймаут или ошибка при чтении системного звука, пропуск блока.\n"
                        )
                        debug_file.flush()
                        continue
                    if mic_stream is not None and mic_result is None:
                        debug_file.write(
                            "   Таймаут или ошибка при чтении микрофона, пропуск блока.\n"
                        )
                        debug_file.flush()
                        continue

                    # Обработка результатов (PyAudio может вернуть tuple или bytes)
                    if isinstance(sys_result, tuple):
                        sys_data_block, sys_overflow = sys_result
                    else:
                        sys_data_block, sys_overflow = sys_result, False
                    if isinstance(mic_result, tuple):
                        mic_data_block, mic_overflow = mic_result
                    else:
                        mic_data_block, mic_overflow = mic_result, False

                    if sys_overflow or mic_overflow:
                        debug_file.write("   Переполнение буфера, блок пропущен.\n")
                        continue

                    # Преобразуем байты в float32
                    sys_block = np.frombuffer(sys_data_block, dtype=np.float32).reshape(
                        -1, channels_sys
                    )
                    mic_block = np.frombuffer(mic_data_block, dtype=np.float32).reshape(
                        -1, channels_mic
                    )

                    # Сохраняем последние аудиоданные для VU-метра
                    with self.recent_audio_lock:
                        self.recent_audio_buffer = mic_block

                    # Усредняем системный звук до моно
                    sys_mono = np.mean(sys_block, axis=1)
                    # Усредняем микрофон до моно
                    mic_mono_raw = np.mean(mic_block, axis=1)

                    # Запись сырых данных микрофона (если включено)
                    if raw_mic_enabled and raw_mic_file:
                        # Записываем сырые float32 данные
                        raw_mic_file.write(mic_mono_raw.astype(np.float32).tobytes())

                    # Ресемплинг
                    if sys_resampler:
                        sys_mono = sys_resampler.process(sys_mono, sys_ratio)
                    if mic_resampler:
                        mic_mono = mic_resampler.process(mic_mono_raw, mic_ratio)
                    else:
                        mic_mono = mic_mono_raw

                    # Приводим к одинаковой длине
                    min_len = min(len(sys_mono), len(mic_mono))
                    if min_len == 0:
                        continue
                    sys_mono = sys_mono[:min_len]
                    mic_mono = mic_mono[:min_len]

                    # Нормализация уровня звука перед микшированием
                    # Получаем параметры нормализации из конфигурации
                    normalization_enabled = (
                        False  # отключаем нормализацию из-за проблем с перегрузом
                    )
                    target_peak = 0.5
                    # Параметры усиления микрофона
                    mic_gain_db = 0.0  # без усиления
                    mic_max_gain = 100.0

                    # Диагностика уровней сигналов перед нормализацией
                    sys_peak_raw = np.max(np.abs(sys_mono)) if len(sys_mono) > 0 else 0
                    mic_peak_raw = np.max(np.abs(mic_mono)) if len(mic_mono) > 0 else 0
                    sys_rms_raw = (
                        np.sqrt(np.mean(sys_mono**2)) if len(sys_mono) > 0 else 0
                    )
                    mic_rms_raw = (
                        np.sqrt(np.mean(mic_mono**2)) if len(mic_mono) > 0 else 0
                    )

                    # Логирование уровней: первые 10 блоков, каждые 500 блоков, при превышении порогов (>1.0)
                    log_levels = (
                        (block_counter < 10)
                        or (block_counter % 500 == 0)
                        or (sys_peak_raw > 1.0)
                        or (mic_peak_raw > 1.0)
                    )
                    if log_levels:
                        debug_file.write(
                            f"   Блок {block_counter}: Sys peak={sys_peak_raw:.6f}, Mic peak={mic_peak_raw:.6f}, "
                            f"Sys RMS={sys_rms_raw:.6f}, Mic RMS={mic_rms_raw:.6f}, Mic gain={mic_gain_db} dB\n"
                        )

                    if normalization_enabled:
                        # Применяем предварительное усиление микрофона
                        if mic_gain_db != 0:
                            mic_gain_linear = 10 ** (mic_gain_db / 20)
                            # Ограничиваем максимальное усиление
                            mic_gain_linear = min(mic_gain_linear, mic_max_gain)
                            mic_mono = mic_mono * mic_gain_linear
                            if log_levels:
                                mic_peak_after_gain = (
                                    np.max(np.abs(mic_mono)) if len(mic_mono) > 0 else 0
                                )
                                debug_file.write(
                                    f"   Блок {block_counter}: Усиление микрофона {mic_gain_db} dB ({mic_gain_linear:.1f}x). Mic peak после усиления={mic_peak_after_gain:.6f}\n"
                                )

                        # Нормализуем системный звук по пиковому значению
                        sys_mono_normalized = normalize_audio_peak(
                            sys_mono, target_peak=target_peak
                        )
                        # Нормализуем микрофон по пиковому значению с увеличенным максимальным усилением
                        mic_mono_normalized = normalize_audio_peak(
                            mic_mono, target_peak=target_peak, max_gain=mic_max_gain
                        )

                        # Диагностика после нормализации
                        if log_levels:
                            sys_peak_norm = (
                                np.max(np.abs(sys_mono_normalized))
                                if len(sys_mono_normalized) > 0
                                else 0
                            )
                            mic_peak_norm = (
                                np.max(np.abs(mic_mono_normalized))
                                if len(mic_mono_normalized) > 0
                                else 0
                            )
                            debug_file.write(
                                f"   Блок {block_counter}: Нормализация. Sys peak={sys_peak_norm:.6f}, Mic peak={mic_peak_norm:.6f}\n"
                            )
                    else:
                        # Если нормализация отключена, используем исходные сигналы
                        sys_mono_normalized = sys_mono
                        mic_mono_normalized = mic_mono
                        if log_levels:
                            debug_file.write(
                                f"   Блок {block_counter}: Нормализация отключена\n"
                            )

                    # Микшируем сигналы (нормализованные или исходные)
                    mixed = (sys_mono_normalized + mic_mono_normalized) * 0.5

                    # Преобразуем в int16
                    mixed_int16 = (mixed * 32767).astype(np.int16)

                    # Отправляем в MP3-энкодер
                    mp3_chunk = encoder.encode(mixed_int16.tobytes())
                    if mp3_chunk:
                        mp3_file.write(mp3_chunk)
                        # Принудительный сброс буфера файловой системы каждые 10 блоков
                        if block_counter % 10 == 0:
                            try:
                                mp3_file.flush()
                                os.fsync(mp3_file.fileno())
                                # Логируем размер файла для диагностики
                                current_size = os.path.getsize(self.mp3_path)
                                debug_file.write(
                                    f"   Сброс буфера, размер файла: {current_size} байт\n"
                                )
                            except Exception as e:
                                debug_file.write(f"   Ошибка при сбросе буфера: {e}\n")

                    block_counter += 1

                    # Логирование прогресса каждые 100 блоков
                    if block_counter % 100 == 0:
                        current_time = time.time()
                        elapsed = current_time - start_time
                        debug_file.write(
                            f"   Прогресс: {block_counter} блоков, время: {elapsed:.1f} сек\n"
                        )

                except Exception as e:
                    debug_file.write(f"   Ошибка в цикле: {e}\n")
                    debug_file.flush()
                    break

            debug_file.write(f"6. Цикл завершён. Записано блоков: {block_counter}\n")

            # --- Финализация MP3 ---
            debug_file.write("   Начало финализации MP3 (flush)...\n")
            mp3_chunk = self._flush_with_timeout(encoder, timeout=2.0)
            if mp3_chunk:
                mp3_file.write(mp3_chunk)
                debug_file.write("   MP3 финализирован успешно.\n")
            else:
                debug_file.write(
                    "   ВНИМАНИЕ: flush завершился с таймаутом или ошибкой, данные могут быть потеряны.\n"
                )

            # --- Закрытие ---
            mp3_file.close()
            # Закрытие файла сырых данных микрофона
            if raw_mic_enabled and raw_mic_file:
                raw_mic_file.close()
                debug_file.write("   Файл сырых данных микрофона закрыт.\n")
            if sys_stream:
                sys_stream.stop_stream()
                sys_stream.close()
            if mic_stream:
                mic_stream.stop_stream()
                mic_stream.close()
            if p:
                p.terminate()

            debug_file.write("=== УСПЕХ ===\n")
            debug_file.close()
            self.on_finish(True, self.mp3_path, None)

        except Exception as e:
            debug_file.write(f"!!! ИСКЛЮЧЕНИЕ: {e}\n")
            traceback.print_exc(file=debug_file)
            debug_file.flush()
            debug_file.close()
            if mp3_file:
                mp3_file.close()
            if raw_mic_enabled and raw_mic_file:
                raw_mic_file.close()
            self.on_finish(False, None, str(e))
