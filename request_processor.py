import os
import json
import requests
import re
from datetime import datetime
from pathlib import Path
import copy

# Импорт функции для получения API ключа из config_manager
from config_manager import get_api_key


def transcribe_audio(audio_path, config):
    url = config["whisper_url"]
    with open(audio_path, "rb") as audio_file:
        files = {"audio_file": audio_file}
        params = {
            "language": config["whisper_language"],
            "word_timestamps": str(config["whisper_word_timestamps"]).lower(),
        }
        headers = {"accept": "application/json"}

        try:
            response = requests.post(url, params=params, files=files, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result
        except Exception as e:
            raise RuntimeError(f"Ошибка при обращении к Whisper ASR: {e}")


def _validate_whisper_json(json_data):
    """Проверяет структуру JSON файла Whisper на соответствие ожидаемому формату."""
    if not isinstance(json_data, dict):
        raise ValueError("JSON должен быть словарём (dict)")

    # Проверяем наличие ключа 'text' или 'segments'
    if "text" not in json_data and "segments" not in json_data:
        raise ValueError("JSON должен содержать ключ 'text' или 'segments'")

    # Если есть 'text', он может быть строкой, списком строк или списком словарей
    if "text" in json_data:
        text = json_data["text"]
        if isinstance(text, list):
            for seg in text:
                # Элемент может быть строкой или словарем с ключом 'text'
                if isinstance(seg, dict):
                    if "text" not in seg:
                        raise ValueError(
                            "Словарь в списке 'text' должен содержать ключ 'text'"
                        )
                elif not isinstance(seg, str):
                    raise ValueError(
                        "Элементы списка 'text' должны быть строками или словарями с ключом 'text'"
                    )
        elif not isinstance(text, str):
            raise ValueError("Ключ 'text' должен быть строкой или списком")

    # Если есть 'segments', проверяем структуру
    if "segments" in json_data:
        segments = json_data["segments"]
        if not isinstance(segments, list):
            raise ValueError("'segments' должен быть списком")
        for seg in segments:
            if not isinstance(seg, dict) or "text" not in seg:
                raise ValueError(
                    "Каждый сегмент в 'segments' должен быть словарём с ключом 'text'"
                )

    return True


def process_json(json_path, config):
    """Загружает и валидирует JSON файл Whisper, возвращает результат для дальнейшей обработки."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Не удалось загрузить JSON файл: {e}")

    # Валидация структуры
    _validate_whisper_json(json_data)

    # Добавляем метаданные, если их нет
    if "audio_file" not in json_data:
        json_data["audio_file"] = os.path.basename(json_path)
    if "timestamp" not in json_data:
        json_data["timestamp"] = datetime.now().isoformat()

    return json_data


def extract_text_from_whisper_result(whisper_result):
    """Извлекает текст из полного ответа Whisper (собирает все сегментов)."""
    # Если есть ключ 'text'
    if "text" in whisper_result:
        text = whisper_result["text"]
        if isinstance(text, list):
            # Список может содержать строки или словари
            result_parts = []
            for seg in text:
                if isinstance(seg, dict):
                    result_parts.append(seg.get("text", ""))
                elif isinstance(seg, str):
                    result_parts.append(seg)
                else:
                    result_parts.append(str(seg))
            return " ".join(result_parts)
        elif isinstance(text, str):
            return text
        else:
            return ""

    # Если есть ключ 'segments'
    if "segments" in whisper_result:
        segments = whisper_result["segments"]
        if isinstance(segments, list):
            return " ".join(seg.get("text", "") for seg in segments)

    # Если структура неизвестна, возвращаем пустую строку
    return ""


def query_llm(prompt_text, asr_text, config):
    """Универсальный запрос к любому OpenAI-совместимому API."""
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "API ключ для LLM не настроен. Откройте Настройки -> LLM и введите ключ."
        )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": config.get("llm_model", "deepseek-chat"),
        "messages": [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": json.dumps(asr_text, ensure_ascii=False)},
        ],
        "temperature": config.get("llm_temperature", 0.7),
    }

    response = requests.post(config["llm_api_url"], json=payload, headers=headers)
    if response.status_code != 200:
        error_text = response.text if response.text else "Пустой ответ"
        raise RuntimeError(
            f"LLM API вернул статус {response.status_code}: {error_text}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"]


def remove_think_tags(text):
    """Удаляет секции <think>...</think> из текста ответа."""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned).strip()
    return cleaned


def save_md_file(content, output_folder, base_name):
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(base_name).stem.replace(" ", "_")
    filename = f"{safe_name}_{timestamp}.md"
    filepath = os.path.join(output_folder, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


def save_whisper_result(audio_path, whisper_result, config):
    folder = config.get("whisper_output_folder", "")
    if not folder:
        folder = os.path.dirname(audio_path)
    os.makedirs(folder, exist_ok=True)

    base_name = Path(audio_path).stem
    filename = f"{base_name}.json"
    filepath = os.path.join(folder, filename)

    if config.get("whisper_remove_words", False):
        result_copy = copy.deepcopy(whisper_result)
        if isinstance(result_copy.get("text"), list):
            for seg in result_copy["text"]:
                if "words" in seg:
                    del seg["words"]
        data = result_copy
    else:
        data = whisper_result

    data["audio_file"] = os.path.basename(audio_path)
    data["timestamp"] = datetime.now().isoformat()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath


def _get_file_type(filepath):
    """Определяет тип файла по расширению."""
    if not filepath:
        return None

    if filepath.lower().endswith(".json"):
        return "json"
    else:
        audio_extensions = {
            ".mp3",
            ".wav",
            ".m4a",
            ".ogg",
            ".flac",
            ".mp4",
            ".mpeg",
            ".opus",
        }
        ext = os.path.splitext(filepath)[1].lower()
        if ext in audio_extensions:
            return "audio"
        else:
            return "unknown"


def start_processing(file_path, prompt_text, config, context_text=""):
    """
    Основная функция обработки файла. Определяет тип файла и маршрутизирует.

    Args:
        file_path: путь к аудиофайлу или JSON файлу Whisper
        prompt_text: текст промпта для LLM
        config: словарь конфигурации
        context_text: дополнительный контекст (опционально)

    Returns:
        tuple: (whisper_result, asr_text, llm_response, saved_md_path)

    Raises:
        ValueError: если тип файла не поддерживается или файл невалиден
        RuntimeError: ошибки обработки
    """
    file_type = _get_file_type(file_path)

    if file_type == "audio":
        # Обработка аудио через Whisper
        whisper_result = transcribe_audio(file_path, config)
        json_path = save_whisper_result(file_path, whisper_result, config)
    elif file_type == "json":
        # Обработка JSON файла
        whisper_result = process_json(file_path, config)
        json_path = file_path  # JSON уже существует
    else:
        raise ValueError(
            f"Неподдерживаемый тип файла: {file_path}\n"
            "Поддерживаются аудиофайлы (.mp3, .wav, .m4a, .ogg, .flac) "
            "и JSON-стенограммы Whisper (.json)"
        )

    # Извлечение текста из результата Whisper
    asr_text = extract_text_from_whisper_result(whisper_result)

    # Подстановка контекста в промпт
    if context_text and "{Context}" in prompt_text:
        prompt_text = prompt_text.replace("{Context}", context_text)

    # Запрос к LLM
    llm_response = query_llm(prompt_text, asr_text, config)
    cleaned_response = remove_think_tags(llm_response)

    # Сохранение результата в MD файл
    base_name = os.path.basename(file_path)
    output_folder = config.get("default_output_folder", os.path.dirname(file_path))
    saved_md_path = save_md_file(cleaned_response, output_folder, base_name)

    return whisper_result, asr_text, llm_response, saved_md_path
