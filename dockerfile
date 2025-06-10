FROM python:3.11-slim

# Install system dependencies including minicom
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    python3-serial \
    build-essential \
    minicom \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install Arduino CLI
RUN curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh && \
    mv bin/arduino-cli /usr/local/bin/

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir -r requirements.txt

# Copy and set permissions on script
COPY scripts/flash_arduino.sh ./scripts/

# Copy application code
COPY . .

# Setup Arduino CLI
RUN arduino-cli core update-index && \
    arduino-cli core install arduino:avr

CMD ["bash"]
