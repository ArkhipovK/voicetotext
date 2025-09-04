# Базовый образ с Python
FROM parakeet-local:latest

# установка дополнительных зависимостей
RUN pip install langchain
RUN pip install langchain-community
RUN pip install librosa
RUN pip install nemo-toolkit

# Устанавливаем рабочую директорию
WORKDIR /app

# Команда запуска
CMD ["python", "app.py"]