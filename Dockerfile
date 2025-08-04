FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    apt-utils \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --no-warn-script-location

COPY . .

RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8080

CMD ["python3", "app.py"]