#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_TOPSCC_NAME "TOPSCC"
#define GGML_TOPSCC_MAX_DEVICES 16

/**
 * @brief Registers the TopsCC backend.
 *
 * This function registers the TopsCC backend with the backend registry.
 *
 * @return A pointer to the backend registration instance.
 */
GGML_API ggml_backend_reg_t ggml_backend_topscc_reg(void);

/**
 * @brief Initializes the TopsCC backend for a specified device.
 *
 * This function initializes the TopsCC backend for the given device.
 * It verifies the device index, allocates a context, and creates a backend
 * instance.
 *
 * @param device The index of the device to initialize.
 * @return A pointer to the initialized backend instance, or nullptr on failure.
 */
GGML_API ggml_backend_t ggml_backend_topscc_init(int device);

/**
 * @brief Checks if a given backend is a TopsCC backend.
 *
 * This function verifies if the provided backend is a TopsCC backend by
 * comparing its GUID with the TopsCC backend's GUID.
 *
 * @param backend The backend instance to check.
 * @return True if the backend is a TopsCC backend, false otherwise.
 */
GGML_API bool ggml_backend_is_topscc(ggml_backend_t backend);

/**
 * @brief Retrieves the TopsCC buffer type for a specified device.
 *
 * This function initializes and returns the buffer type interface associated
 * with the given device. It ensures thread-safe access using a mutex.
 *
 * @param device The device index for which to retrieve the buffer type.
 * @return A pointer to the buffer type interface for the specified device, or
 * nullptr if the device index is out of range.
 */
GGML_API ggml_backend_buffer_type_t ggml_backend_topscc_buffer_type(int device);

/**
 * @brief Retrieves the number of TopsCC devices available.
 *
 * This function returns the number of TopsCC devices available based on
 * information obtained from `ggml_topscc_info()`.
 *
 * @return The number of TopsCC devices available.
 */
GGML_API int32_t ggml_backend_topscc_get_device_count(void);

/**
 * @brief pinned host buffer for use with the CPU backend for faster copies
 * between CPU and GCU.
 *
 * @return A pointer to the host buffer type interface.
 */
GGML_API ggml_backend_buffer_type_t ggml_backend_topscc_host_buffer_type(void);

/**
 * @brief Retrieves the name of a TopsCC device.
 *
 * This function retrieves the name of the TopsCC device with the given index.
 *
 * @param device The index of the device.
 * @param description Pointer to a buffer where the description will be written.
 * @param description_size Size of the description buffer.
 */
GGML_API void
ggml_backend_topscc_get_device_description(int device, char *description,
                                           size_t description_size);

/**
 * @brief Retrieves the memory information of a TopsCC device.
 *
 * @param device The device index to retrieve memory information for.
 * @param free Pointer to a variable where the free memory size will be stored.
 * @param total Pointer to a variable where the total memory size will be
 * stored.
 */
GGML_API void ggml_backend_topscc_get_device_memory(int device, size_t *free,
                                                    size_t *total);

#ifdef __cplusplus
}
#endif
