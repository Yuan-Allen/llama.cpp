#ifndef TOPSCC_COMMON_H
#define TOPSCC_COMMON_H

#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../include/ggml-topscc.h"
#include "../include/ggml.h"

#define MATRIX_ROW_PADDING 512

void ggml_topscc_set_device(int32_t device);

/**
 * @brief Contains information about TOPSCC devices.
 */
struct ggml_topscc_device_info {
    /**
     * @brief Number of TOPSCC devices available.
     */
    int32_t device_count;

    /**
     * @brief Information about a single TOPSCC device.
     */
    struct topscc_device_info {
        // TODO: add device information
    };

    topscc_device_info devices[GGML_TOPSCC_MAX_DEVICES] =
        {}; /**< Array of TOPSCC device information. */
};

struct ggml_backend_topscc_context {
    int32_t device;   /**< Device ID. */
    std::string name; /**< Name of the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_topscc_context(int device)
        : device(device), name("TOPSCC" + std::to_string(device)) {}

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_topscc_context() {
        ggml_topscc_set_device(device);
        // TODO: cleanup resources
    }

    // TODO: add more members
};

#endif // TOPSCC_COMMON_H
