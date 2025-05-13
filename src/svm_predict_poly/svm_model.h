#ifndef SVM_MODEL_H
#define SVM_MODEL_H

#include <Arduino.h>

const int n_classes = 3;
const int n_support_vectors = 8;
const int vector_length = 2;

const float support_vectors[8][2] = {
    {-0.446748, 0.013669},
    {-0.459677, 0.006412},
    {-0.459683, 0.006415},
    {-0.459669, 0.006427},
    {-0.459683, 0.006416},
    {-0.458928, 0.006736},
    {-0.459061, 0.006683},
    {-0.453842, 0.009702}
};
const float dual_coef[2][8] = {
    {0.879183, -0.000000, -0.000000, -0.879183, -0.000000, -0.000000, -0.000000, -3.123187},
    {3.123187, 63.725490, 61.565937, 63.725490, 63.725490, -125.291427, -127.450980, -0.000000}
};
const float intercept[3] = { 11.223850, 20.309426, -197.654621 };
const int class_labels[3] = { 0.000000, 1.000000, 2.000000 };

const float gamma = 10.0000000000;
const float coef0 = 0.000000;
const int degree = 6;

// === Scaler Parameter ===
const float scaler_mean[2] = { 0.000160, 0.097490 };
const float scaler_scale[2] = { 0.000484, 0.303631 };

#endif // SVM_MODEL_H
