FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy gradio huggingface_hub && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir transformers datasets scikit-learn accelerate

COPY . .

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
