import json
from flask import Flask, request, jsonify
import nemo.collections.asr as nemo_asr
import torch
import gc
import time
import logging
import subprocess
import pandas as pd, json
from segment import (
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
    asr_model = nemo_asr.models.ASRModel.restore_from(f"models/asr/{model_name}", map_location=device)
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

            config_path = "config/vad_inference_frame.yaml"
            vad_in_manifest = "/app/config/vad_in_manifest.json"
            # vad_out_manifest = "/app/config/vad_out_manifest.json"

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

            transcripts = []
            transcription_text = ""

            for audio_tensor, meta in dl:
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
        with open("/tmp/output.txt", "w") as f:
            f.write(f"{transcription_text}")

        print("transcripts")
        for tr in transcripts:
            print(tr)

        with open("/tmp/output.json", "w", encoding="utf-8") as f:
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
