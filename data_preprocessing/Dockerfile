# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app


# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install compatible versions of opencv-python-headless and numpy
RUN pip install --no-cache-dir numpy==1.23.5 opencv-python-headless==4.5.5.64

# Copy the Python script
COPY rd_video_image_extractor.py .

# Set the entrypoint
ENTRYPOINT ["python", "rd_video_image_extractor.py"]
