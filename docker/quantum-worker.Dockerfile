FROM python:3.11
WORKDIR /app
COPY python_quantum/requirements.txt .
RUN pip install -r requirements.txt
COPY python_quantum .
CMD ["python", "worker.py"]