#include <stdarg.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-topscc.h"

#define GGML_COMMON_DECL_C

#include "ggml-common.h"
#include "ggml-topscc/common.h"
#include "ggml-topscc/topscc_ops.h"

#define GGML_TOPSCC_NAME "TOPSCC"

void ggml_topscc_set_device(int device) {
    // TODO: set device
}

static ggml_topscc_device_info ggml_topscc_init() {
    ggml_topscc_device_info info = {};

    info.device_count = 1; // TODO: get device count

    GGML_ASSERT(info.device_count <= GGML_TOPSCC_MAX_DEVICES);

    // TODO: add more device info later.
    return info;
}

const ggml_topscc_device_info &ggml_topscc_info() {
    static ggml_topscc_device_info info = ggml_topscc_init();
    return info;
}

// topscc buffer

struct ggml_backend_topscc_buffer_context {
    int device;
    void *dev_ptr = nullptr;
    std::string name;

    ggml_backend_topscc_buffer_context(int device, void *dev_ptr)
        : device(device), dev_ptr(dev_ptr),
          name(GGML_TOPSCC_NAME + std::to_string(device)) {}

    ~ggml_backend_topscc_buffer_context() {
        // TODO: free dev_ptr
    }
};

static void
ggml_backend_topscc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_topscc_buffer_context *ctx =
        (ggml_backend_topscc_buffer_context *)buffer->context;
    delete ctx;
}

static void *ggml_backend_topscc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_topscc_buffer_context *ctx =
        (ggml_backend_topscc_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static void
ggml_backend_topscc_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                         ggml_tensor *tensor, uint8_t value,
                                         size_t offset, size_t size) {
    ggml_backend_topscc_buffer_context *ctx =
        (ggml_backend_topscc_buffer_context *)buffer->context;

    ggml_topscc_set_device(ctx->device);
    // TODO: memset tensor
    memset((char *)tensor->data + offset, value, size);
}

static void ggml_backend_topscc_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                  ggml_tensor *tensor,
                                                  const void *data,
                                                  size_t offset, size_t size) {
    ggml_backend_topscc_buffer_context *ctx =
        (ggml_backend_topscc_buffer_context *)buffer->context;

    ggml_topscc_set_device(ctx->device);
    // TODO: set tensor
    memcpy((char *)tensor->data + offset, data, size);
}

static void ggml_backend_topscc_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                  const ggml_tensor *tensor,
                                                  void *data, size_t offset,
                                                  size_t size) {
    ggml_backend_topscc_buffer_context *ctx =
        (ggml_backend_topscc_buffer_context *)buffer->context;

    ggml_topscc_set_device(ctx->device);
    // TODO: get tensor
    memcpy(data, (const char *)tensor->data + offset, size);
}

static void ggml_backend_topscc_buffer_clear(ggml_backend_buffer_t buffer,
                                             uint8_t value) {
    ggml_backend_topscc_buffer_context *ctx =
        (ggml_backend_topscc_buffer_context *)buffer->context;

    ggml_topscc_set_device(ctx->device);
    // TODO: clear buffer
    memset(ctx->dev_ptr, value, buffer->size);
}

static const ggml_backend_buffer_i ggml_backend_topscc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_topscc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_topscc_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_topscc_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_topscc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_topscc_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_topscc_buffer_clear,
    /* .reset           = */ NULL,
};

// topscc buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
struct ggml_backend_topscc_buffer_type_context {
    int32_t
        device; /**< Device identifier associated with the buffer context. */
    std::string name; /**< Name associated with the buffer context. */
};

/**
 * @brief Retrieves the name associated with a TOPSCC buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * TOPSCC buffer type context.
 *
 * @param buft Pointer to the buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
static const char *
ggml_backend_topscc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_topscc_buffer_type_context *buft_ctx =
        (ggml_backend_topscc_buffer_type_context *)buft->context;

    return buft_ctx->name.c_str();
}

/**
 * @brief Checks if the backend buffer type is associated with the TOPSCC
 * backend.
 *
 * This function checks whether the provided backend buffer type is associated
 * with the TOPSCC backend based on the comparison of its name retrieval
 * function pointer.
 *
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the buffer type is associated with the TOPSCC
 * backend, otherwise false.
 */
static bool ggml_backend_buft_is_topscc(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_topscc_buffer_type_name;
}

static ggml_backend_buffer_t
ggml_backend_topscc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                             size_t size) {
    ggml_backend_topscc_buffer_type_context *buft_ctx =
        (ggml_backend_topscc_buffer_type_context *)buft->context;

    ggml_topscc_set_device(buft_ctx->device);

    void *dev_ptr;
    // TODO: allocate device memory
    dev_ptr = malloc(size);

    ggml_backend_topscc_buffer_context *ctx =
        new ggml_backend_topscc_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_topscc_buffer_interface,
                                    ctx, size);
}

static size_t
ggml_backend_topscc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // TODO: choose the correct alignment
    return 1;

    GGML_UNUSED(buft);
}

static size_t
ggml_backend_topscc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                               const ggml_tensor *tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING -
                                                    ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

/**
 * @brief Interface for managing TOPSCC buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for TOPSCC buffer types in the GGML backend.
 */
static const ggml_backend_buffer_type_i
    ggml_backend_topscc_buffer_type_interface = {
        /* .get_name         = */ ggml_backend_topscc_buffer_type_name,
        /* .alloc_buffer     = */ ggml_backend_topscc_buffer_type_alloc_buffer,
        /* .get_alignment    = */ ggml_backend_topscc_buffer_type_get_alignment,
        /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
        /* .get_alloc_size   = */
        ggml_backend_topscc_buffer_type_get_alloc_size,
        /* .is_host          = */ NULL,
};

/**
 * @brief Retrieves the TOPSCC buffer type for a specified device.
 *
 * This function initializes and returns the buffer type interface associated
 * with the given device. It ensures thread-safe access using a mutex.
 *
 * @param device The device index for which to retrieve the buffer type.
 * @return A pointer to the buffer type interface for the specified device, or
 * nullptr if the device index is out of range.
 */
ggml_backend_buffer_type_t ggml_backend_topscc_buffer_type(int32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_topscc_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type
        ggml_backend_topscc_buffer_types[GGML_TOPSCC_MAX_DEVICES];

    static bool ggml_backend_topscc_buffer_type_initialized = false;

    if (!ggml_backend_topscc_buffer_type_initialized) {
        for (int32_t i = 0; i < GGML_TOPSCC_MAX_DEVICES; i++) {
            ggml_backend_topscc_buffer_types[i] = {
                /* .iface    = */ ggml_backend_topscc_buffer_type_interface,
                /* .device    = */
                ggml_backend_reg_dev_get(ggml_backend_topscc_reg(), device),
                /* .context  = */
                new ggml_backend_topscc_buffer_type_context{
                    i, "TOPSCC" + std::to_string(i)},
            };
        }
        ggml_backend_topscc_buffer_type_initialized = true;
    }

    return &ggml_backend_topscc_buffer_types[device];
}

static bool ggml_topscc_compute_forward(ggml_backend_topscc_context &ctx,
                                        struct ggml_tensor *dst) {
    switch (dst->op) {
    // TODO: support more operations
    case GGML_OP_MUL_MAT:
        ggml_topscc_mul_mat(ctx, dst->src[0], dst->src[1], dst);
        break;
    default:
        return false;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char *ggml_backend_topscc_get_name(ggml_backend_t backend) {
    ggml_backend_topscc_context *topscc_ctx =
        (ggml_backend_topscc_context *)backend->context;

    return topscc_ctx->name.c_str();
}

static void ggml_backend_topscc_free(ggml_backend_t backend) {
    ggml_backend_topscc_context *topscc_ctx =
        (ggml_backend_topscc_context *)backend->context;

    delete topscc_ctx;
    delete backend;
}

static enum ggml_status
ggml_backend_topscc_graph_compute(ggml_backend_t backend, ggml_cgraph *cgraph) {
    ggml_backend_topscc_context *topscc_ctx =
        (ggml_backend_topscc_context *)backend->context;

    ggml_topscc_set_device(topscc_ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) {
            continue;
        }

        bool ok = ggml_topscc_compute_forward(*topscc_ctx, node);

        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__,
                           node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Structure defining the interface for the TOPSCC backend.
 *
 * This structure contains function pointers for various operations
 * supported by the TOPSCC backend, including name retrieval, memory
 * management, tensor operations, synchronization, and event handling.
 *
 * TODO: implement these functions
 */
static const ggml_backend_i ggml_backend_topscc_interface = {
    /* .get_name                = */ ggml_backend_topscc_get_name,
    /* .free                    = */ ggml_backend_topscc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_topscc_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

/**
 * @brief Interface for managing TOPSCC host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for TOPSCC buffer types in the GGML backend.
 */
ggml_backend_buffer_type_t ggml_backend_topscc_host_buffer_type() {
    static struct ggml_backend_buffer_type
        // TODO: implement these functions
        ggml_backend_topscc_buffer_type_host = {
            /* .iface    = */ {
                /* .get_name         = */
                NULL,
                /* .alloc_buffer     = */
                NULL,
                /* .get_alignment    = */
                ggml_backend_cpu_buffer_type()->iface.get_alignment,
                /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
                /* .get_alloc_size   = */
                ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                /* .is_host          = */
                ggml_backend_cpu_buffer_type()->iface.is_host,
            },
            /* .device   = */
            ggml_backend_reg_dev_get(ggml_backend_topscc_reg(), 0),
            /* .context  = */ nullptr,
        };

    return &ggml_backend_topscc_buffer_type_host;
}

/**
 * @brief Checks if the TOPSCC backend supports a specific operation.
 *
 * This function checks whether the specified operation is supported by the
 * TOPSCC backend.
 *
 * @param backend Pointer to the TOPSCC backend structure to check support for
 *                the operation.
 * @param op Pointer to the tensor representing the operation to check.
 * @return bool Returns true if the operation is supported by the backend,
 *              otherwise false.
 */
static bool ggml_backend_topscc_device_supports_op(ggml_backend_dev_t dev,
                                                   const ggml_tensor *op) {
    switch (op->op) {
    // TODO: support more operations
    // case GGML_OP_MUL_MAT:
    //     return true;
    default:
        return false;
    }

    GGML_UNUSED(dev);
}

/**
 * @brief Determines if a tensor operation should be offloaded to the TOPSCC
 * backend.
 *
 * This function checks if a given tensor operation should be offloaded to the
 * TOPSCC backend based on the operation type and the size of the tensor. It
 * returns true if the second dimension (ne[1]) of the tensor is greater than or
 * equal to the minimum batch size and the operation is not GGML_OP_GET_ROWS.
 *
 * @param backend Pointer to the TOPSCC backend.
 * @param op Pointer to the tensor operation to check.
 * @return bool Returns true if the operation should be offloaded, otherwise
 * false.
 */
static bool ggml_backend_topscc_device_offload_op(ggml_backend_dev_t dev,
                                                  const ggml_tensor *op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);

    // TODO: implement this function
    return false;
}

/**
 * @brief Return the hardcoded GUID for the TOPSCC backend.
 *
 * This function returns a static GUID which uniquely identifies the TOPSCC
 * backend.
 *
 * @return A pointer to the static GUID.
 */
static ggml_guid_t ggml_backend_topscc_guid() {
    static ggml_guid guid = {0x22, 0xf9, 0xc5, 0x99, 0x19, 0x89, 0x44, 0x96,
                             0x2e, 0xec, 0x5f, 0x85, 0xf6, 0xc9, 0x5b, 0x19};
    return &guid;
}

// backend device
struct ggml_backend_topscc_device_context {
    int device;
    std::string name;
    std::string description;
};

static const char *ggml_backend_topscc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_topscc_device_context *ctx =
        (ggml_backend_topscc_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char *
ggml_backend_topscc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_topscc_device_context *ctx =
        (ggml_backend_topscc_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_topscc_device_get_memory(ggml_backend_dev_t dev,
                                                  size_t *free, size_t *total) {
    ggml_backend_topscc_device_context *ctx =
        (ggml_backend_topscc_device_context *)dev->context;
    ggml_backend_topscc_get_device_memory(ctx->device, free, total);
}

static enum ggml_backend_dev_type
ggml_backend_topscc_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void
ggml_backend_topscc_device_get_props(ggml_backend_dev_t dev,
                                     ggml_backend_dev_props *props) {
    props->name = ggml_backend_topscc_device_get_name(dev);
    props->description = ggml_backend_topscc_device_get_description(dev);
    props->type = ggml_backend_topscc_device_get_type(dev);
    ggml_backend_topscc_device_get_memory(dev, &props->memory_free,
                                          &props->memory_total);

    bool host_buffer = getenv("GGML_TOPSCC_NO_PINNED") == nullptr;

    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_topscc_device_init(ggml_backend_dev_t dev,
                                                      const char *params) {
    GGML_UNUSED(params);
    ggml_backend_topscc_device_context *ctx =
        (ggml_backend_topscc_device_context *)dev->context;
    return ggml_backend_topscc_init(ctx->device);
}

/**
 * @brief Checks if the TOPSCC backend supports a specific backend buffer type.
 *
 * This function determines whether the TOPSCC backend supports the given
 * backend buffer type by comparing the device context of the backend and buffer
 * type. It returns true if the devices are same between the backend context and
 * buffer type context.
 *
 * @param backend Pointer to the TOPSCC backend.
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the TOPSCC backend supports the buffer type,
 *              otherwise false.
 */
static bool
ggml_backend_topscc_device_supports_buft(ggml_backend_dev_t dev,
                                         ggml_backend_buffer_type_t buft) {
    if (ggml_backend_buft_is_topscc(buft)) {
        ggml_backend_topscc_device_context *dev_ctx =
            (ggml_backend_topscc_device_context *)dev->context;
        ggml_backend_topscc_buffer_type_context *buft_ctx =
            (ggml_backend_topscc_buffer_type_context *)buft->context;
        return buft_ctx->device == dev_ctx->device;
    }
    return false;
}

static ggml_backend_buffer_type_t
ggml_backend_topscc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_topscc_device_context *ctx =
        (ggml_backend_topscc_device_context *)dev->context;
    return ggml_backend_topscc_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t
ggml_backend_topscc_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_topscc_host_buffer_type();
}

// TODO: implement these functions
static const ggml_backend_device_i ggml_backend_topscc_device_interface = {
    /* .get_name                = */ ggml_backend_topscc_device_get_name,
    /* .get_description         = */ ggml_backend_topscc_device_get_description,
    /* .get_memory              = */ ggml_backend_topscc_device_get_memory,
    /* .get_type                = */ ggml_backend_topscc_device_get_type,
    /* .get_props               = */ ggml_backend_topscc_device_get_props,
    /* .init_backend            = */ ggml_backend_topscc_device_init,
    /* .get_buffer_type         = */ ggml_backend_topscc_device_get_buffer_type,
    /* .get_host_buffer_type    = */
    ggml_backend_topscc_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_topscc_device_supports_op,
    /* .supports_buft           = */ ggml_backend_topscc_device_supports_buft,
    /* .offload_op              = */ ggml_backend_topscc_device_offload_op,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

// backend reg

struct ggml_backend_topscc_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char *ggml_backend_topscc_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_TOPSCC_NAME;
}

static size_t ggml_backend_topscc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_topscc_reg_context *ctx =
        (ggml_backend_topscc_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t
ggml_backend_topscc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_topscc_reg_context *ctx =
        (ggml_backend_topscc_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

// TODO: implement these functions
static const ggml_backend_reg_i ggml_backend_topscc_reg_interface = {
    /* .get_name          = */ ggml_backend_topscc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_topscc_reg_get_device_count,
    /* .get_device_get    = */ ggml_backend_topscc_reg_get_device,
    /* .get_proc_address  = */ NULL,
};

// backend registry, called only once for topscc backend
ggml_backend_reg_t ggml_backend_topscc_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_topscc_reg_context *ctx =
                new ggml_backend_topscc_reg_context;

            for (int i = 0; i < ggml_topscc_info().device_count; i++) {
                ggml_backend_topscc_device_context *dev_ctx =
                    new ggml_backend_topscc_device_context();
                dev_ctx->description = "GCU"; // TODO: get device description
                dev_ctx->device = i;
                dev_ctx->name = GGML_TOPSCC_NAME + std::to_string(i);
                ggml_topscc_set_device(i);
                ggml_backend_dev_t dev = new ggml_backend_device{
                    /* .interface = */ ggml_backend_topscc_device_interface,
                    /* .reg       = */ &reg,
                    /* .context   = */ dev_ctx};
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{
                /* .interface = */ ggml_backend_topscc_reg_interface,
                /* .context   = */ ctx};
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_topscc_init(int32_t device) {
    if (device < 0 || device >= ggml_backend_topscc_get_device_count()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_topscc_context *ctx = new ggml_backend_topscc_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    }
    ggml_topscc_set_device(ctx->device);
    ggml_backend_t topscc_backend = new ggml_backend{
        /* .guid      = */ ggml_backend_topscc_guid(),
        /* .interface = */ ggml_backend_topscc_interface,
        /* .device    = */
        ggml_backend_reg_dev_get(ggml_backend_topscc_reg(), device),
        /* .context   = */ ctx};

    return topscc_backend;
}

bool ggml_backend_is_topscc(ggml_backend_t backend) {
    return backend != NULL &&
           ggml_guid_matches(backend->guid, ggml_backend_topscc_guid());
}

int32_t ggml_backend_topscc_get_device_count() {
    return ggml_topscc_info().device_count;
}

void ggml_backend_topscc_get_device_description(int32_t device,
                                                char *description,
                                                size_t description_size) {
    ggml_topscc_set_device(device);
    const char *soc_name = "GCU"; // TODO: get soc name
    snprintf(description, description_size, "%s", soc_name);
}

void ggml_backend_topscc_get_device_memory(int32_t device, size_t *free,
                                           size_t *total) {
    ggml_topscc_set_device(device);
    // TODO: get device memory
}
