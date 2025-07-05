# Use the official Python 3.11 slim image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# [Fix] Install system dependencies required by LightGBM and other ML libraries.
# libgomp.1 is required for OpenMP support.
# postgresql-client is needed for the db-init service to use psql.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 postgresql-client && rm -rf /var/lib/apt/lists/*

# To leverage Docker's layer caching, copy only the dependency file first
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project's source code into the working directory
COPY . .

# Expose the port the API server will run on
EXPOSE 8000
