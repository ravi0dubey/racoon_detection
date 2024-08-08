# Use the voxel51/fiftyone:0.24.1 image as the base
FROM voxel51/fiftyone:0.24.1 AS build-stage

# Set the working directory
WORKDIR /app

# Install necessary packages and clean up in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies in a single layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Create a directory for images
RUN mkdir -p /app/images

# Set the default command
CMD ["python", "deduplicate_images.py"]
