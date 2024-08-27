# Dockerfile
FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the app with watchfiles for hot reloading
CMD ["python3", "src/app.py"]
