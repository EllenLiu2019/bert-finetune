FROM python:3.8-slim

WORKDIR /app

COPY deploy/requirements.txt /app
RUN pip install --upgrade pip -r requirements.txt

COPY src/infer_service.py /app

EXPOSE 8000

CMD ["uvicorn", "infer_service:app", "--host", "0.0.0.0", "--port", "8000"]
