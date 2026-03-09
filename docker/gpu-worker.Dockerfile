FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /app
COPY python_quantum/requirements.txt .
RUN pip3 install -r requirements.txt
COPY gpu_worker.py .
CMD ["python3", "gpu_worker.py"]