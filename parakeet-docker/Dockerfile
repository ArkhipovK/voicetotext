# Базовый образ с Python
FROM python:3.10-slim

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y \
    wget git ffmpeg build-essential gcc g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Обновление pip и установка PyTorch с поддержкой CPU
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Установка pybind11 перед texterrors
RUN pip install pybind11

# Копируем файл зависимостей
COPY requirements.txt .

# Установка зависимостей без изоляции сборки
RUN pip install --use-pep517 --no-build-isolation -r requirements.txt

# # Копируем файлы приложения и модель
# COPY app.py /app/app.py
# COPY parakeet-tdt-0.6b-v2.nemo /app/parakeet-tdt-0.6b-v2.nemo

# Устанавливаем рабочую директорию
WORKDIR /app

# Команда запуска
CMD ["python", "app.py"]