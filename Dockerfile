FROM python:3.11-slim

WORKDIR /app

# Copy your code
# from /mnt/e/Dev/Otus/crypto-robot-cloud/
COPY . .

# Copy requirements and install
# COPY requirements.txt .
RUN pip install -r requirements.txt


# Optional: set environment variables
ENV PYTHONUNBUFFERED=1

# If you have a main script for the model serving:
CMD ["python", "ensemble_dashboard.py"]
