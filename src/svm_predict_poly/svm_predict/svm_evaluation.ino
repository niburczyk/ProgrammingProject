#pragma once

#include "svm_model.h"

#define VEC_DIM vector_length
#define GAMMA gamma
#define COEF0 coef0
#define DEGREE degree

#include "poly_kernel.ino"

float* normalize_input(float input[]) {
    static float normalized[vector_length];
    for (int i = 0; i < vector_length; i++) {
        normalized[i] = (input[i] - scaler_mean[i]) / scaler_scale[i];
    }
    return normalized;
}

int svm_predict(const float* input) {
    float decision[n_classes] = {0.0};

    for (int i = 0; i < n_classes; ++i) {
        for (int j = 0; j < n_support_vectors; ++j) {
            float k = rbf_kernel((const float*)support_vectors[j], input);
            decision[i] += dual_coef[i][j] * k;
        }
        decision[i] += intercept[i];
    }

    int predicted_class = 0;
    float max_decision = decision[0];
    for (int i = 1; i < n_classes; ++i) {
        if (decision[i] > max_decision) {
            max_decision = decision[i];
            predicted_class = i;
        }
    }

    return class_labels[predicted_class];
}
