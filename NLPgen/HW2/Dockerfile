FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04
RUN \
    apt update && \
    apt install -yf python3 python3-pip

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

CMD ["python3", "./app.py"]
