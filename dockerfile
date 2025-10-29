FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure internal cache folder exists
RUN mkdir -p /app/cache

CMD ["python", "app.py"]
