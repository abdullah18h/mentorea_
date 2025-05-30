FROM python:3.11.12-slim

WORKDIR /app

# Install build dependencies for lightfm
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install gunicorn explicitly
RUN pip install --no-cache-dir gunicorn
# Debug: Verify gunicorn installation and list files
RUN which gunicorn && gunicorn --version && ls -la

# Copy application code
COPY . .

# Debug: Verify app.py exists
RUN ls -la app.py || echo "app.py not found"

# Use shell form for CMD to allow $PORT substitution
CMD gunicorn --bind 0.0.0.0:$PORT --log-level debug app:app
