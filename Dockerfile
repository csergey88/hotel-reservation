FROM python:slim

ENV PYTHONWRITEDBYTECODE = 1
ENV PYTHONUNBUFFERED = 1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgomp1  \
    && rm -rf /var/lib/apt/lists/*

COPY . .


RUN pip install --upgrade pip
RUN pip install -e . --no-cache-dir


RUN python pipeline/training_pipelione.py

EXPOSE 5001

CMD ["python", "application.py"]






