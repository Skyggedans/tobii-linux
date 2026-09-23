/* tobii/tobii_wearable.h — head-mounted devices. Declared so wearable-aware
 * code builds; nothing here applies to the screen-based ET5, and every entry
 * point returns TOBII_ERROR_NOT_SUPPORTED. Companion to tobii/tobii.h.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef TOBII_WEARABLE_H
#define TOBII_WEARABLE_H

#include "tobii.h"
#include "tobii_streams.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tobii_wearable_eye_t
{
    tobii_validity_t pupil_position_in_sensor_area_validity;
    float pupil_position_in_sensor_area_xy[ 2 ];
    tobii_validity_t position_guide_validity;
    float position_guide_xy[ 2 ];
    tobii_validity_t blink_validity;
    tobii_state_bool_t blink;
} tobii_wearable_eye_t;

typedef struct tobii_wearable_consumer_data_t
{
    int64_t timestamp_us;
    tobii_wearable_eye_t left;
    tobii_wearable_eye_t right;
    tobii_validity_t gaze_origin_combined_validity;
    float gaze_origin_combined_mm_xyz[ 3 ];
    tobii_validity_t gaze_direction_combined_validity;
    float gaze_direction_combined_normalized_xyz[ 3 ];
    tobii_validity_t convergence_distance_validity;
    float convergence_distance_mm;
    tobii_state_bool_t improve_user_position_hmd;
    tobii_state_bool_t increase_eye_relief;
} tobii_wearable_consumer_data_t;

typedef void ( *tobii_wearable_consumer_data_callback_t )(
    tobii_wearable_consumer_data_t const* data, void* user_data );

typedef struct tobii_wearable_advanced_data_t
{
    int64_t timestamp_tracker_us;
    int64_t timestamp_system_us;
    tobii_wearable_eye_t left;
    tobii_wearable_eye_t right;
    tobii_validity_t gaze_origin_combined_validity;
    float gaze_origin_combined_mm_xyz[ 3 ];
    tobii_validity_t gaze_direction_combined_validity;
    float gaze_direction_combined_normalized_xyz[ 3 ];
    tobii_validity_t convergence_distance_validity;
    float convergence_distance_mm;
    tobii_state_bool_t improve_user_position_hmd;
    tobii_state_bool_t increase_eye_relief;
} tobii_wearable_advanced_data_t;

typedef void ( *tobii_wearable_advanced_data_callback_t )(
    tobii_wearable_advanced_data_t const* data, void* user_data );

typedef struct tobii_lens_configuration_t
{
    float left_xyz[ 3 ];
    float right_xyz[ 3 ];
} tobii_lens_configuration_t;

typedef enum tobii_lens_configuration_writable_t
{
    TOBII_LENS_CONFIGURATION_NOT_WRITABLE,
    TOBII_LENS_CONFIGURATION_WRITABLE,
} tobii_lens_configuration_writable_t;

typedef enum tobii_wearable_foveated_tracking_state_t
{
    TOBII_WEARABLE_FOVEATED_TRACKING_STATE_TRACKING,
    TOBII_WEARABLE_FOVEATED_TRACKING_STATE_EXTRAPOLATED,
    TOBII_WEARABLE_FOVEATED_TRACKING_STATE_LAST_KNOWN,
} tobii_wearable_foveated_tracking_state_t;

typedef struct tobii_wearable_foveated_gaze_t
{
    int64_t timestamp_us;
    tobii_wearable_foveated_tracking_state_t tracking_state;
    float gaze_direction_combined_normalized_xyz[ 3 ];
} tobii_wearable_foveated_gaze_t;

typedef void ( *tobii_wearable_foveated_gaze_callback_t )(
    tobii_wearable_foveated_gaze_t const* data, void* user_data );

TOBII_API tobii_error_t TOBII_CALL tobii_wearable_consumer_data_subscribe( tobii_device_t* device,
    tobii_wearable_consumer_data_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_wearable_consumer_data_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_wearable_advanced_data_subscribe( tobii_device_t* device,
    tobii_wearable_advanced_data_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_wearable_advanced_data_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_get_lens_configuration( tobii_device_t* device,
    tobii_lens_configuration_t* lens_config );
TOBII_API tobii_error_t TOBII_CALL tobii_set_lens_configuration( tobii_device_t* device,
    tobii_lens_configuration_t const* lens_config );
TOBII_API tobii_error_t TOBII_CALL tobii_lens_configuration_writable( tobii_device_t* device,
    tobii_lens_configuration_writable_t* writable );
TOBII_API tobii_error_t TOBII_CALL tobii_wearable_foveated_gaze_subscribe( tobii_device_t* device,
    tobii_wearable_foveated_gaze_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_wearable_foveated_gaze_unsubscribe( tobii_device_t* device );

#ifdef __cplusplus
}
#endif

#endif /* TOBII_WEARABLE_H */
