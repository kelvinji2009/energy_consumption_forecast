# Use the official Python 3.11 slim image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# To leverage Docker's layer caching, copy only the dependency file first
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project's source code into the working directory
COPY . .

# Expose the port the API server will run on
EXPOSE 8000
