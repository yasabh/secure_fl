# Base image
FROM python:3.10.16-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    vim \
    gcc \
    g++ \
    rsync \
    unzip \
    cython3 \
    python3-dev \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    && apt-get clean \
    && apt-get autoclean

# Set working directory
WORKDIR /app

# Copy SAFEFL repository into container
COPY . /app

# Install MP-SPDZ
RUN bash mpc_install.sh