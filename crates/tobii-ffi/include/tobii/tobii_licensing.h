/* tobii/tobii_licensing.h — licensing, as provided by libtobii.so. There is
 * nothing to license: every key validates and the feature group is consumer.
 * Companion to tobii/tobii.h.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef TOBII_LICENSING_H
#define TOBII_LICENSING_H

#include "tobii.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tobii_license_key_t
{
    uint16_t const* license_key;
    size_t size_in;
} tobii_license_key_t;

typedef enum tobii_license_validation_result_t
{
    TOBII_LICENSE_VALIDATION_RESULT_OK,
    TOBII_LICENSE_VALIDATION_RESULT_TAMPERED,
    TOBII_LICENSE_VALIDATION_RESULT_INVALID_APPLICATION_SIGNATURE,
    TOBII_LICENSE_VALIDATION_RESULT_NONSIGNED_APPLICATION,
    TOBII_LICENSE_VALIDATION_RESULT_EXPIRED,
    TOBII_LICENSE_VALIDATION_RESULT_PREMATURE,
    TOBII_LICENSE_VALIDATION_RESULT_INVALID_PROCESS_NAME,
    TOBII_LICENSE_VALIDATION_RESULT_INVALID_SERIAL_NUMBER,
    TOBII_LICENSE_VALIDATION_RESULT_INVALID_MODEL,
    TOBII_LICENSE_VALIDATION_RESULT_INVALID_PLATFORM_TYPE,
} tobii_license_validation_result_t;

typedef enum tobii_feature_group_t
{
    TOBII_FEATURE_GROUP_BLOCKED,
    TOBII_FEATURE_GROUP_CONSUMER,
    TOBII_FEATURE_GROUP_CONFIG,
    TOBII_FEATURE_GROUP_PROFESSIONAL,
    TOBII_FEATURE_GROUP_INTERNAL,
} tobii_feature_group_t;

/* tobii_device_create; every key is reported OK without being read. */
TOBII_API tobii_error_t TOBII_CALL tobii_device_create_ex( tobii_api_t* api, char const* url,
    tobii_field_of_use_t field_of_use, tobii_license_key_t const* license_keys,
    int license_count, tobii_license_validation_result_t* license_results,
    tobii_device_t** device );
/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED */
TOBII_API tobii_error_t TOBII_CALL tobii_license_key_store( tobii_device_t* device,
    void* data, size_t size );
/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED */
TOBII_API tobii_error_t TOBII_CALL tobii_license_key_retrieve( tobii_device_t* device,
    tobii_data_receiver_t receiver, void* user_data );
/* TOBII_FEATURE_GROUP_CONSUMER (gaze data is served anyway). */
TOBII_API tobii_error_t TOBII_CALL tobii_get_feature_group( tobii_device_t* device,
    tobii_feature_group_t* feature_group );

#ifdef __cplusplus
}
#endif

#endif /* TOBII_LICENSING_H */
