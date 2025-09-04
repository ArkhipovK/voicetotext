import json
from flask import Flask, request, jsonify
import nemo.collections.asr as nemo_asr
import torch
import gc
import os
import sys
import time
import logging
import subprocess
import difflib
import soundfile as sf
import librosa
import numpy as np
import pandas as pd, json
from pathlib import Path 
from segment import (
    segment_audio,
    frame_segment_audio,
    SegmentDataset
)
from torch.utils.data import DataLoader

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from nemo.collections.asr.parts.utils.vad_utils import (
    prepare_manifest,
)
from nemo.collections.asr.parts.utils.manifest_utils import (
    create_manifest
)


app = Flask(__name__)

# Увеличиваем максимальный размер файла до 2 ГБ
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 ГБ

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Определяем устройство (CPU или MPS для Mac M1/M2)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Использую MPS для Apple Silicon")
else:
    device = torch.device("cpu")
    print("Использую CPU")

# Загрузка модели из локального файла
print("Загрузка модели...")
# model_name = "parakeet-tdt-0.6b-v2.nemo"
model_name = "stt_ru_conformer_ctc_large.nemo"
try:
    logger.info(f"Загрузка модели {model_name}...")
    asr_model = nemo_asr.models.ASRModel.restore_from(model_name, map_location=device)
    # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name).cpu()   
    # asr_model.eval()
    # logger.info(f"Модель загружена на {device}")
    # asr_model = EncDecHybridRNNTCTCModel.restore_from(model_name, map_location=device)
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/restart", methods=['GET'])
def restart():
    subprocess.run("shutdown -r 0", shell=True, check=True)
    return "Restarting"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Логируем начало запроса
    print("Получен запрос на транскрипцию")
    gc.collect()
    file = request.files.get('file')
    if not file:
        print("Ошибка: файл не найден в запросе")
        return jsonify({"error": "Файл не найден"}), 400

    # Сохраняем файл на диск
    file_path = "./audio.wav"
    output_path = "./audio/prepared/sample.wav"
    try:
        file.save(file_path)
        logger.info(f"Файл успешно сохранен: {file_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении файла: {e}")
        return jsonify({"error": "Ошибка при сохранении файла"}), 500

    try:
        logger.info(f"преобразование в нужный формат: {file_path}")
        file_path =  auto_convert_audio(file_path, output_path)

        logger.info(f"Начало транскрипции файла: {file_path}")
        start_time = time.time()

        # Выполняем транскрипцию
        with torch.no_grad():
            # segments = nemo_segment_audio(
            #     file_path,
            #     segment_duration=10.0,
            #     overlap=1.5,
            #     target_sr=16000
            # )
            logger.info(os.getcwd())
            # config_path = "config/vad_inference_postprocessing.yaml"
            config_path = "config/vad_inference_frame.yaml"
            vad_in_manifest = "/app/config/vad_in_manifest.json"
            vad_out_manifest = "/app/config/vad_out_manifest.json"

            create_manifest(
                wav_path = "config/wav_paths.txt",
                manifest_filepath = vad_in_manifest
            )
          
            frame_segment_audio(
                vad_in_manifest,
                config_path,
                output_dir = "output/vad_frame"
            )

            segTable = pd.read_csv("output/vad_frame/rttm_preds/sample.txt", delim_whitespace=True, names=["start", "duration", "label"])
            segTable = segTable[segTable["label"] == "speech"]     # keep only speech

           

            audio_queries = []
            for _, row in segTable.iterrows():
                audio_queries.append({
                    # "audio_filepath": "/app/prepared/sample.wav",  # original long file
                    "offset": float(row["start"]),
                    "duration": float(row["duration"])
                })
            ds = SegmentDataset(file_path, audio_queries)
            dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

            # for start_sec, end_sec, segment in segments:
            #     print(f"Обработка сегмента: {start_sec:.1f}-{end_sec:.1f} сек")
                
            #     # Получение сэмплов
            #     samples = segment.samples
            #     print(np.array(samples).shape)
               
            #     # print(np.array(samples).shape)
            #     # Преобразование в тензор [batch, time]
            #     audio_tensor = torch.tensor(samples).unsqueeze(0).float()

            #        # Convert to NumPy array
            #     # numpy_array = tensor_data.numpy()

            #     # Now, you can serialize the NumPy array (or a dictionary containing it)
            #     data_to_serialize = {"my_tensor_data": samples.tolist()} # Convert to list for JSON
                
            #     return jsonify(data_to_serialize)

            #     # audio_tensor = prepare_audio_tensor(samples)

                
            #     # Транскрипция
            #     text = asr_model.transcribe(audio_tensor)[0]
                
            #     # Сохранение результата с временными метками
            #     transcripts.append({
            #         "start": start_sec,
            #         "end": end_sec,
            #         "text": text
            #     })
                
            #     # Очистка памяти
            #     del audio_tensor, samples
            #     torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            transcripts = []
            transcription_text = ""

            for audio_tensor, meta in dl:
                # audio_tensor: shape [1, N_samples], dtype float32
                audio_np = audio_tensor.squeeze(0).numpy()
     
                hyps = asr_model.transcribe(
                    audio_np, 
                    batch_size=1,
                    return_hypotheses=False,
                    timestamps=True
                )

                for hyp in hyps:
                    abstract = []
                    total = len(hyp.timestamp["segment"]) - 1
                    for i, s in enumerate(hyp.timestamp["segment"]):
                        abstract.append({
                            "start": s["start"],
                            "duration":   s["end"],
                            "text":  s["segment"]
                        })
                        if meta["duration"].tolist()[0] < 2:
                             transcription_text += s["segment"] + " "
                        else:
                            transcription_text += capitalize_text(s["segment"], i == total)
                    transcripts.append({
                        "offset" : meta["offset"].tolist()[0],
                        "duration" : meta["duration"].tolist()[0],
                        "abstract": abstract
                    })
                    if meta["duration"].tolist()[0] >= 2:
                        transcription_text += "\n"
        # Логируем время обработки
        logger.info(f"Транскрипция завершена за {time.time()-start_time:.2f} сек")

        # Логируем результат
        with open("/app/output/output.txt", "w") as f:
            f.write(f"{transcription_text}")

        print("transcripts")
        for tr in transcripts:
            print(tr)

        with open("/app/output/output.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(transcripts, ensure_ascii=False))
        print("Результат записан в output.json")
        # return jsonify({"transcription": str(transcription_text)})
        return jsonify(transcripts)

    except Exception as e:
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.exception(str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/qa', methods=['POST'])
def qa():
    query = request.form.get('query')
    # 1. Загрузка текста
    with open("/tmp/output.txt") as f:
        text = f.read()

    # 2. Чанкование + векторизация
    embedder = HuggingFaceEmbeddings(model_name="sbert_large_nlu_ru")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    vector_db = Qdrant.from_texts(chunks, embedder, location=":memory:")

    # 3. Настройка QA-цепи
    llm = HuggingFacePipeline.from_model_id(
        model_id="IlyaGusev/saiga2_7b_lora",
        task="text-generation",
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=vector_db.as_retriever(),
        chain_type="stuff"  # или "map_reduce" для очень длинных текстов
    )

    # 4. Задаём вопрос
    print(qa.run(query))

def nemo_segment_audio(
    file_path, 
    segment_duration=30.0, 
    overlap=1.0, 
    target_sr=16000
):
    """
    Сегментирует аудиофайл с преобразованием в моно и понижением частоты
    
    :param file_path: путь к аудиофайлу
    :param segment_duration: длительность сегмента в секундах
    :param overlap: перекрытие между сегментами в секундах
    :param target_sr: целевая частота дискретизации
    :return: список кортежей (начало, конец, AudioSegment)
    """
    # Загрузка аудио с конвертацией в моно
    signal, orig_sr = sf.read(file_path)
    
    # Преобразование в моно
    if signal.ndim > 1:
        signal = signal.squeeze()  # Удаление избыточной размерности
    
    # Понижающая дискретизация
    if orig_sr != target_sr:
        signal = librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)
        sample_rate = target_sr
    else:
        sample_rate = orig_sr
    
    # Рассчет параметров сегментации
    total_duration = len(signal) / sample_rate
    segment_samples = int(segment_duration * sample_rate)
    step_samples = int((segment_duration - overlap) * sample_rate)
    
    # Создание сегментов
    segments = []
    start_sample = 0
    
    while start_sample < len(signal):
        end_sample = start_sample + segment_samples
        
        # Обработка последнего сегмента
        if end_sample > len(signal):
            end_sample = len(signal)
        
        # Извлечение сегмента сигнала
        segment_signal = signal[start_sample:end_sample]
        
        # Создание AudioSegment
        segment = AudioSegment(
            samples=segment_signal,
            sample_rate=sample_rate,
            target_sr=target_sr,
            duration=len(segment_signal) / sample_rate,
            offset=start_sample / sample_rate,
            orig_sr=orig_sr
        )
        
        # Добавление в результат
        segments.append((
            start_sample / sample_rate,  # начало в секундах
            end_sample / sample_rate,    # конец в секундах
            segment
        ))
        
        # Переход к следующему сегменту
        start_sample += step_samples
        
        # Остановка если достигнут конец
        if start_sample >= len(signal):
            break
    
    return segments


    # signal, sr = librosa.load(file_path, sr=16000, mono=True)
    
    # Создание временного манифеста
    manifest = [{
        "audio_filepath": file_path,
        "duration": librosa.get_duration(file_path),
        "text": ""  # Текст не нужен для транскрипции
    }]

    # Создание сегментов
    segmented_manifest = create_segments(
        manifest=manifest,
        segment_duration=60.0,
        step_duration=58.5,  # 60 - 1.5 overlap
        min_segment_duration=1.0,
        output_dir="temp_segments",
    )

    for entry in segmented_manifest:
        segment_path = entry[file_path]
        transcription = asr_model.transcribe([segment_path])[0]
        transcripts.append(transcription)

        
        for i, seg in enumerate(segments):
            # Добавление размерности батча
            seg_tensor = torch.tensor(seg).unsqueeze(0)
            transcripts = []

            # transcription = asr_model.transcribe(seg_tensor, channel_selector=0)
            if seg_tensor.dim() == 3:
                print("Конвертация стерео в моно...")
                seg_tensor = seg_tensor.mean(dim=2) if seg_tensor.size(2) > 1 else seg_tensor[:, :, 0]
            
            text = model.transcribe(seg_tensor)[0]
            transcripts.append(text)
            # Очистка памяти
            # del seg_tensor
            # torch.cuda.empty_cache() if torch.cuda.is_available() else Non
        return " ".join(transcripts)

def auto_convert_audio(file_path, output_path):
    """Конвертирует любой формат в WAV 16kHz mono"""

    # name = Path(file_path).name
    try:
        subprocess.run([
            "ffmpeg", "-i", file_path,
            "-ac", "1",          # моно
            "-ar", "16000",      # 16kHz
            "-acodec", "pcm_s16le",
            "-f", "wav",
            "-loglevel", "error",
            "-vn",
            "-y",                # перезаписать если существует
            output_path,
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown FFmpeg error"
        raise RuntimeError(f"Ошибка конвертации: {error_msg}") from e
    
    return output_path

def combine_segments_with_overlap(results, min_overlap_words=3, max_overlap_words=5, similarity_threshold=0.8):
    """
    Склеивает результаты транскрипции сегментов с учетом перекрытия
    
    :param results: список словарей [{"start": float, "end": float, "text": str}, ...]
    :param min_overlap_words: минимальное количество слов для поиска в перекрытии
    :param max_overlap_words: максимальное количество слов для поиска в перекрытии
    :param similarity_threshold: порог схожести для совпадения фраз
    :return: склеенный текст
    """
    if not results:
        return ""
    
    full_text = results[0]['text']
    prev_text = full_text
    
    for i in range(1, len(results)):
        current = results[i]
        current_text = current['text']
        
        # Разбиваем тексты на слова
        prev_words = prev_text.split()
        curr_words = current_text.split()
        
        # Определяем диапазон поиска перекрытия
        search_range = range(min_overlap_words, min(max_overlap_words, len(prev_words), len(curr_words)) + 1)
        
        # Поиск оптимального перекрытия
        best_match = None
        best_similarity = 0
        
        for n in search_range:
            if n > len(prev_words) or n > len(curr_words):
                continue
                
            # Берем последние n слов предыдущего текста
            prev_overlap = " ".join(prev_words[-n:])
            # Берем первые n слов текущего текста
            curr_overlap = " ".join(curr_words[:n])
            
            # Рассчитываем схожесть
            similarity = difflib.SequenceMatcher(
                None, prev_overlap.lower(), curr_overlap.lower()
            ).ratio()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (n, prev_overlap, curr_overlap)
        
        # Обработка найденного перекрытия
        if best_match and best_similarity >= similarity_threshold:
            n, prev_phrase, curr_phrase = best_match
            # Добавляем только неперекрывающуюся часть
            remaining_text = " ".join(curr_words[n:])
            
            # Для отладки можно выводить информацию о склейке
            print(f"🔗 Обнаружено перекрытие {n} слов: "
                  f"«{prev_phrase}» → «{curr_phrase}» "
                  f"(сходство: {best_similarity:.2f})")
            
            full_text += " " + remaining_text
        else:
            # Если перекрытие не найдено, добавляем весь текст
            full_text += " " + current_text
            if best_match:
                print(f"⚠️ Низкое сходство ({best_similarity:.2f}): "
                      f"«{best_match[1]}» vs «{best_match[2]}»")
        
        prev_text = current_text
    
    return full_text.strip()

def prepare_audio_tensor(audio_data):
    """
    Преобразует аудио данные в правильный формат для модели
    :param audio_data: аудио данные (numpy array или tensor)
    :return: тензор в формате [batch, time]
    """
    # Преобразование в тензор PyTorch, если необходимо
    if not isinstance(audio_data, torch.Tensor):
        audio_tensor = torch.tensor(audio_data)
    else:
        audio_tensor = audio_data
    
    # Проверка и исправление размерности
    if audio_tensor.dim() == 1:  # [time]
        audio_tensor = audio_tensor.unsqueeze(0)  # [1, time]
    elif audio_tensor.dim() == 3:  # [batch, channels, time]
        # Убираем размерность каналов
        audio_tensor = audio_tensor.squeeze(1)  # [batch, time]
    elif audio_tensor.dim() == 2 and audio_tensor.size(0) != 1:
        # Если размерность [time, channels], транспонируем
        audio_tensor = audio_tensor.transpose(0, 1)
        audio_tensor = audio_tensor.squeeze(1)  # [batch, time]
    
    return audio_tensor.float()

def capitalize_text(text, last_sent = False):
    if not text.strip():  # Проверка на пустую строку или пробелы
        return ". "
    cleaned = text.rstrip()  # Удаление пробелов в конце

    end_space =  " " if not last_sent else ""
    # Добавление точки, если её нет, и пробела
    result = cleaned + '.' * (not cleaned.endswith('.')) + end_space
    # Поиск первой буквы для преобразования в заглавную
    for i, char in enumerate(result):
        if char.isalpha():
            return result[:i] + char.upper() + result[i+1:]
    return result  # Возврат без изменений, если букв нет

if __name__ == '__main__':
    # Запускаем сервер Flask с поддержкой многопоточности
    print("Запуск сервера Flask на 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
