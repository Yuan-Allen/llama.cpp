#ifndef GGML_TOPSCC_OPS
#define GGML_TOPSCC_OPS

#include "common.h"

void ggml_topscc_mul_mat(ggml_backend_topscc_context &ctx,
                         const ggml_tensor *src0, const ggml_tensor *src1,
                         ggml_tensor *dst);

#endif // GGML_TOPSCC_OPS