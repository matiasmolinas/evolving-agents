# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Set the working directory in the container
WORKDIR ${APP_HOME}

# Install system dependencies that might be needed by Python packages
# RUN apt-get update && apt-get install -y --no-install-recommends #     build-essential #  && rm -rf /var/lib/apt/lists/*
# Note: For now, let's assume requirements.txt handles pre-built wheels or simple builds.

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code from the host to the container
COPY . .

# (Optional) Add a non-root user
# RUN addgroup --system app && adduser --system --ingroup app app
# USER app

# Default command can be omitted if docker-compose specifies it or if using exec
