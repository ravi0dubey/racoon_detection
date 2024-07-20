# Use the official Python image from the Docker Hub
FROM python:3.9-slim
# Set the working directory
WORKDIR /app
# Copy the requirements file
COPY requirements.txt .
# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the application code
COPY . .
# Set environment variables
# ENV GOOGLE_APPLICATION_CREDENTIALS=gcp_service_account.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service_account.json

