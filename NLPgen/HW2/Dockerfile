FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./app.py"]
