FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app/

RUN apt-get update -y && \
    apt-get install -y awscli && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]