# Use a Python image that supports TensorFlow
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy your project files
COPY . .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Start the Flask app
CMD ["python", "app.py"]