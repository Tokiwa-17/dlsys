//
// Created by ylf on 2023/5/28.
//
#include <iostream>
#include <cmath>
#include <gtest/gtest.h>

float * Iy(const unsigned char *y, int l, int r, size_t batch, size_t k) {
    float * _Iy = new float(batch * k);
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < k; j++) {
            if (j == y[l + i])
                _Iy[i * k + j] = 1.0f;
            else _Iy[i * k + j] = 0.0f;
        }
    }
    return _Iy;
}

float *multiply(const float *a, const float *b, int a_height, int a_width, int b_height, int b_width) {
    if (a_width != b_height) return nullptr;
    float *c = new float(a_height * b_width);
    for (int i = 0; i < a_height; i++)
        for (int j = 0; j < b_height; j++) {
            c[i * b_width + j] = 0.0f;
            for (int k = 0; k < a_width; k++) {
                c[i * b_width + j] += a[i * a_width + k] * b[k * b_width + j];
            }
        }
    return c;
}

float *exp(const float *a, int a_height, int a_width) {
    auto b = new float(a_height * a_width);
    for (int i = 0; i < a_height; i++)
        for (int j = 0; j < a_width; j++) {
            b[i * a_width + j] = std::expf(a[i * a_width + j]);
        }
    return b;
}

TEST(MULTIPLYTEST, Positive) {
    float* a = new float[4] {0.0f, 1.0f, 2.0f, 3.0f};
    float* b = new float[4] {0.0f, 1.0f, 2.0f, 3.0f};
    auto c = multiply(a, b, 2, 2, 2, 2);
    EXPECT_EQ(c[0], 2);
    EXPECT_EQ(c[1], 3);
    EXPECT_EQ(c[2], 6);
    EXPECT_EQ(c[3], 11);
    delete[] a;
    delete[] b;
    delete[] c;
}

TEST(EXPTEST, Positive) {
    float * a = new float[4] {0.0f, 1.0f, 2.0f, 3.0f};
    auto b = exp(a, 2, 2);
    float * res = new float[4] { 1.        ,  2.71828183,  7.3890561 , 20.08553692};
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(b[i], res[i], 1e-4);
    }
    delete[] a;
    delete[] b;
    delete[] res;
}


// main 函数，运行测试用例
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
