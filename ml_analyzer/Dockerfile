FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml_analyzer.py .
CMD ["python", "ml_analyzer.py"]
