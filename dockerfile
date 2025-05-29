FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl unzip python3-serial build-essential

# Arduino CLI
RUN curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh && \
    mv bin/arduino-cli /usr/local/bin/

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN arduino-cli core update-index && \
    arduino-cli core install arduino:avr

CMD ["bash"]
