#include "topscc_ops.h"

void ggml_topscc_mul_mat(ggml_backend_topscc_context &ctx,
                         const ggml_tensor *src0, const ggml_tensor *src1,
                         ggml_tensor *dst) {
    // Fake implementation for testing purposes
    int rows = 4;
    int cols = 3;
    float fake_matrix[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};

    // Assuming dst is a 2D tensor with the same dimensions
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ((float *)dst->data)[i * cols + j] = fake_matrix[i][j];
        }
    }
}
