import os
import threading
import numpy as np
import pyaudiowpatch as pyaudio
import traceback
import lameenc
import samplerate
from datetime import datetime
from pathlib import Path


# Функции для ресемплинга аудио
def create_resampler(mode="sinc_best", channels=1):
    return samplerate.Resampler(mode, channels=channels)


def resample_audio(data, ratio, resampler=None):
    """Ресемплит аудио данные"""
    return resampler.process(data, ratio)


# ==================== НОРМАЛИЗАЦИЯ АУДИО ====================
def normalize_audio(data, target_level_db=-3.0, eps=1e-8):
    """
    Нормализует аудиосигнал до заданного уровня в dBFS.

    Args:
        data: numpy array с аудиоданными (float32)
        target_level_db: целевой уровень в dBFS (по умолчанию -3 dB)
        eps: маленькое значение для избежания деления на ноль

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
        gain_linear = 10 ** (gain_db / 20)

        # Применяем усиление с ограничением (максимум +20 dB)
        max_gain = 10  # +20 dB в линейной шкале
        gain_linear = np.clip(gain_linear, 0, max_gain)

        return data * gain_linear
    else:
        # Сигнал слишком тихий или нулевой
        return data


def normalize_audio_peak(data, target_peak=0.9, eps=1e-8):
    """
    Нормализует аудиосигнал по пиковому значению.

    Args:
        data: numpy array с аудиоданными (float32)
        target_peak: целевое пиковое значение (0.0 - 1.0)
        eps: маленькое значение для избежания деления на ноль

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

        # Ограничиваем максимальное усиление (максимум +20 dB ~ 10x)
        max_gain = 10
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

    def _get_whisper_json_path(self, audio_path):
        """Возвращает ожидаемый путь к JSON-файлу Whisper для данного аудио."""
        folder = self.config.get("whisper_output_folder", "")
        if not folder:
            folder = os.path.dirname(audio_path)
        base_name = Path(audio_path).stem
        return os.path.join(folder, f"{base_name}.json")

    def start_recording(self, save_path, blocksize=4096):
        self.recording = True
        self.mp3_path = save_path
        self.blocksize = blocksize
        self.thread = threading.Thread(target=self._record)
        self.thread.daemon = True
        self.thread.start()

    def stop_recording(self):
        self.recording = False

    def _record(self):
        log_dir = os.path.join(os.path.dirname(self.mp3_path), "log")
        os.makedirs(log_dir, exist_ok=True)
        debug_filename = os.path.join(
            log_dir, f"debug_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        debug_file = open(debug_filename, "w", encoding="utf-8")
        debug_file.write("=== НАЧАЛО ЗАПИСИ (ПОТОКОВЫЙ РЕСЕМПЛИНГ) ===\n")

        p = None
        sys_stream = None
        mic_stream = None
        mp3_file = None
        encoder = None
        try:
            debug_file.write("1. Инициализация PyAudio...\n")
            p = pyaudio.PyAudio()

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

            # Поиск loopback
            loopback_device = None
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    loopback_device = loopback
                    debug_file.write(f"   Найдено loopback: {loopback['name']}\n")
                    break
            if loopback_device is None:
                raise RuntimeError("Не найдено устройство захвата системного звука.")

            # Микрофон
            mic_info = p.get_default_input_device_info()
            debug_file.write(f"   Микрофон: {mic_info['name']}\n")

            # Параметры устройств
            fs_sys = int(loopback_device["defaultSampleRate"])
            fs_mic = int(mic_info["defaultSampleRate"])
            channels_sys = loopback_device["maxInputChannels"]
            channels_mic = mic_info["maxInputChannels"]
            debug_file.write(f"   Системный звук: {fs_sys} Гц, {channels_sys} канала\n")
            debug_file.write(f"   Микрофон: {fs_mic} Гц, {channels_mic} канала\n")

            # Целевая частота – максимальная из двух
            target_fs = max(fs_sys, fs_mic)
            debug_file.write(f"   Целевая частота: {target_fs} Гц\n")

            # --- Открываем потоки ---
            debug_file.write("3. Открытие потоков ввода...\n")
            sys_stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels_sys,
                rate=fs_sys,
                frames_per_buffer=self.blocksize,
                input=True,
                input_device_index=loopback_device["index"],
            )
            mic_stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels_mic,
                rate=fs_mic,
                frames_per_buffer=self.blocksize,
                input=True,
                input_device_index=mic_info["index"],
            )
            debug_file.write("   Потоки открыты.\n")

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

            mp3_file = open(self.mp3_path, "wb")
            debug_file.write(f"   MP3-файл открыт: {self.mp3_path}\n")

            # --- Цикл записи ---
            debug_file.write("5. Начало цикла записи...\n")
            block_counter = 0

            while self.recording:
                try:
                    sys_result = sys_stream.read(
                        self.blocksize, exception_on_overflow=False
                    )
                    mic_result = mic_stream.read(
                        self.blocksize, exception_on_overflow=False
                    )

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

                    # Усредняем системный звук до моно
                    sys_mono = np.mean(sys_block, axis=1)
                    # Усредняем микрофон до моно
                    mic_mono_raw = np.mean(mic_block, axis=1)

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
                    normalization_enabled = self.config.get(
                        "audio_normalization_enabled", True
                    )
                    target_peak = self.config.get(
                        "audio_normalization_peak_target", 0.5
                    )

                    if normalization_enabled:
                        # Нормализуем системный звук по пиковому значению
                        sys_mono_normalized = normalize_audio_peak(
                            sys_mono, target_peak=target_peak
                        )
                        # Нормализуем микрофон по пиковому значению
                        mic_mono_normalized = normalize_audio_peak(
                            mic_mono, target_peak=target_peak
                        )

                        # Отладочная информация о уровнях (только для первых нескольких блоков)
                        if block_counter < 5:
                            sys_peak = (
                                np.max(np.abs(sys_mono)) if len(sys_mono) > 0 else 0
                            )
                            mic_peak = (
                                np.max(np.abs(mic_mono)) if len(mic_mono) > 0 else 0
                            )
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
                                f"   Блок {block_counter}: Нормализация включена. Sys peak={sys_peak:.4f}->{sys_peak_norm:.4f}, Mic peak={mic_peak:.4f}->{mic_peak_norm:.4f}\n"
                            )
                    else:
                        # Если нормализация отключена, используем исходные сигналы
                        sys_mono_normalized = sys_mono
                        mic_mono_normalized = mic_mono
                        if block_counter < 5:
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

                    block_counter += 1

                except Exception as e:
                    debug_file.write(f"   Ошибка в цикле: {e}\n")
                    break

            debug_file.write(f"6. Цикл завершён. Записано блоков: {block_counter}\n")

            # --- Финализация MP3 ---
            mp3_chunk = encoder.flush()
            if mp3_chunk:
                mp3_file.write(mp3_chunk)
            debug_file.write("   MP3 финализирован.\n")

            # --- Закрытие ---
            mp3_file.close()
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
            debug_file.close()
            if mp3_file:
                mp3_file.close()
            self.on_finish(False, None, str(e))
