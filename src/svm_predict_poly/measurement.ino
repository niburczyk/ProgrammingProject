#pragma once

#include "svm_evaluation.ino"

const int analogInPin = A0;
const unsigned long sampleInterval = 1;
const int windowSize = 100;

float sensorData[windowSize];
int dataIndex = 0;
float sum = 0.0;

float mav = 0.0;
float wl = 0.0;

unsigned long lastSampleTime = 0;
float restAverage = 0;

bool paused = false;
bool measuring = false;

unsigned long restSum = 0;
unsigned int restCount = 0;
unsigned long startTime = 0;
const unsigned long restDuration = 10000;

void initMeasurement() {
    startTime = millis();
    for (int i = 0; i < windowSize; i++) {
        sensorData[i] = 0;
    }
    sum = 0;
    dataIndex = 0;
    restSum = 0;
    restCount = 0;
    measuring = false;
    paused = false;
}

float calculate_mav() {
    return sum / windowSize;
}

float calculate_wl() {
    float sum_wl = 0.0;
    for (int i = 1; i < windowSize; i++) {
        sum_wl += abs(sensorData[i] - sensorData[i - 1]);
    }
    return sum_wl;
}

void performMeasurementStep() {
    unsigned long currentTime = millis();

    if (!measuring && (currentTime - startTime <= restDuration)) {
        if (currentTime - lastSampleTime >= sampleInterval) {
            lastSampleTime = currentTime;
            int value = analogRead(analogInPin);
            restSum += value;
            restCount++;
        }
    } else if (!measuring) {
        restAverage = (float)restSum / restCount;
        measuring = true;
        Serial.println("Messung beginnt...");
    } else if (!paused && (currentTime - lastSampleTime >= sampleInterval)) {
        lastSampleTime = currentTime;

        int rawValue = analogRead(analogInPin);
        float diff = rawValue - restAverage;

        sum += diff - sensorData[dataIndex];
        sensorData[dataIndex] = diff;
        dataIndex = (dataIndex + 1) % windowSize;

        mav = calculate_mav();
        wl = calculate_wl();

        float x_input[vector_length] = {mav, wl};
        int prediction = svm_predict(normalize_input(x_input));

        Serial.print("Vorhergesagte Klasse: ");
        Serial.println(prediction);
    }
}
