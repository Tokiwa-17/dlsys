#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;

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

float *normalize(const float *a, int a_height, int a_width) {
    auto b = new float(a_height * a_width);
    std::vector<float> _sum;
    for (int i = 0; i < a_height; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < a_width; j++) {
            row_sum += a[i * a_width + j];
            b[i * a_width + j] = a[i * a_width + j];
        }
        _sum.emplace_back(row_sum);
    }
    for (int i = 0; i < a_height; i++) {
        for (int j = 0; j < a_width; j++) {
            b[i * a_width + j] /= _sum[i];
        }
    }
    return b;
}

float * generateIy(const unsigned char *y, int l, int r, int height, int width) {
    float * Iy = new float(height * width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            Iy[i * width + j] = 0.0f;
        }
    }
    for (int i = 0; i < height; i++) {
        int j = y[l + i];
        Iy[i * width + j] = 1.0f;
    }
    return Iy;
}

void subtract(float *a, float *b, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            a[i * width + j] -= b[i * width + j];
        }
    }
}

float * transpose(const float *a, int height, int width) {
    float *b = new float[height * width];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            b[j * height + i] = a[i * width + j];
        }
    }
    return b;
}

void elementMultiply(float *a, float val, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            a[i * width + j] *= val;
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int iter = std::ceil(m / batch);
    for (int i = 0; i < iter; i++) {
        auto l = i * batch, r = (i + 1) * batch >= m ? m : (i + 1) * batch;
        // X[l:r, :] @ theta
        auto tmp1 = multiply(X + l * n, theta, r - l + 1, n, n, k);
        // np.exp(X[l:r, :] @ theta
        auto tmp2 = exp(tmp1, r - l + 1, k);
        // normalize(np.exp(X[l:r, :] @ theta)
        auto tmp3 = normalize(tmp2, r - l + 1, k);
        // Iy = np.eye(k)[y[l:r]]
        auto Iy = generateIy(y, l, r, r - l + 1, k);
        // (normalize(np.exp(X[l:r, :] @ theta) - Iy)
        subtract(tmp3, Iy, r - l + 1, k);
        // X[l:r, :].T
        auto tmp4 = transpose(X + l * n, r - l + 1, n);
        // X[l:r, :].T @ (normalize(np.exp(X[l:r, :] @ theta)) - Iy)
        auto tmp5 = multiply(tmp4, tmp3, n, r - l + 1, r - l + 1, k);
        //  lr * (1 / batch) * X[l:r, :].T @ (normalize(np.exp(X[l:r, :] @ theta)) - Iy)
        elementMultiply(tmp5, lr * 1 / (r - l + 1), n, k);
        // result
        subtract(theta, tmp5, n, k);
        delete[] tmp1;
        delete[] tmp2;
        delete[] tmp3;
        delete[] tmp4;
        delete[] tmp5;
        delete[] Iy;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
