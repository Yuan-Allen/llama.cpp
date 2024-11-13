# Llama

## Related OPs

- `GGML_OP_GET_ROWS`
    - e.g. `void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
- `GGML_OP_SCALE`
    - Multiply the tensor by a scalar.
    - e.g. `void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
- `GGML_OP_CPY`
    - e.g. `void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1)`
- `GGML_OP_NORM`
    - e.g. `void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
- `GGML_OP_RMS_NORM`
    - e.g. `void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
- `GGML_OP_MUL`
    - e.g. `void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
- `GGML_OP_ADD`
    - e.g. `void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
- `GGML_OP_MUL_MAT`
    - e.g. `static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst)`
- `GGML_OP_ROPE`
    - Rotary Position Embedding，RoPE.
    - e.g. `void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
- `GGML_OP_RESHAPE`
    - Nothing to do when computing forward.
- `GGML_OP_VIEW`
    - Nothing to do when computing forward.
- `GGML_OP_TRANSPOSE`
    - Nothing to do when computing forward.
- `GGML_OP_PERMUTE`
    - Nothing to do when computing forward.
- `GGML_OP_UNARY`
    - When we have a unary operation, we need to switch on the operation and call the appropriate function.
    - e.g. 
    ```c++
    static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
            // ...
            case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                // other unary ops...
                default:
                    return false;
            }
            break;
            // ...
    }
    ```
- `GGML_UNARY_OP_TANH`
    - e.g. `void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
- `GGML_OP_SOFT_MAX`
    - e.g. `void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
- `GGML_OP_CONT`
    - e.g. `void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
    - It then calls `ggml_cuda_cpy`: 
    ```c++
    void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
        const ggml_tensor * src0 = dst->src[0];
        ggml_cuda_cpy(ctx, src0, dst);
    }
    ```
- `GGML_UNARY_OP_SILU`
    - e.g. `void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`
- `GGML_UNARY_OP_RELU`
    - e.g. `void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst)`

## Caution

The Matrix multiplication in `llama.cpp` is weird. See [this](../../CONTRIBUTING.md) for more details.
