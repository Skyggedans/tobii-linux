/* tobii/tobii_advanced.h — per-eye gaze data, as provided by libtobii.so.
 * Companion to tobii/tobii.h and tobii/tobii_streams.h.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef TOBII_ADVANCED_H
#define TOBII_ADVANCED_H

#include "tobii.h"
#include "tobii_streams.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tracker frame, mm (origin at the tracker, z towards the user). The ET5
 * reports no pupil diameter: pupil_validity is always INVALID. */
typedef struct tobii_gaze_data_eye_t
{
    tobii_validity_t gaze_origin_validity;
    float gaze_origin_from_eye_tracker_mm_xyz[ 3 ];
    float gaze_origin_in_track_box_normalized_xyz[ 3 ];

    tobii_validity_t gaze_point_validity;
    float gaze_point_from_eye_tracker_mm_xyz[ 3 ];
    float gaze_point_on_display_normalized_xy[ 2 ];

    tobii_validity_t eyeball_center_validity;
    float eyeball_center_from_eye_tracker_mm_xyz[ 3 ];

    tobii_validity_t pupil_validity;
    float pupil_diameter_mm;
} tobii_gaze_data_eye_t;

/* timestamp_tracker_us is on the tracker's clock; timestamp_system_us is the
 * same instant on the tobii_system_clock clock, as the DLL gives it, not the
 * time the sample was received. */
typedef struct tobii_gaze_data_t
{
    int64_t timestamp_tracker_us;
    int64_t timestamp_system_us;
    tobii_gaze_data_eye_t left;
    tobii_gaze_data_eye_t right;
} tobii_gaze_data_t;

typedef void ( *tobii_gaze_data_callback_t )( tobii_gaze_data_t const* gaze_data, void* user_data );

TOBII_API tobii_error_t TOBII_CALL tobii_gaze_data_subscribe( tobii_device_t* device,
    tobii_gaze_data_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_gaze_data_unsubscribe( tobii_device_t* device );

typedef void ( *tobii_digital_syncport_callback_t )( uint32_t signal, int64_t timestamp_tracker_us,
    int64_t timestamp_system_us, void* user_data );

/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED */
TOBII_API tobii_error_t TOBII_CALL tobii_digital_syncport_subscribe( tobii_device_t* device,
    tobii_digital_syncport_callback_t callback, void* user_data );
/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED */
TOBII_API tobii_error_t TOBII_CALL tobii_digital_syncport_unsubscribe( tobii_device_t* device );

typedef char tobii_face_type_t[ 64 ];
typedef void ( *tobii_face_type_receiver_t )( tobii_face_type_t const face_type, void* user_data );

/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED */
TOBII_API tobii_error_t TOBII_CALL tobii_enumerate_face_types( tobii_device_t* device,
    tobii_face_type_receiver_t receiver, void* user_data );
/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED */
TOBII_API tobii_error_t TOBII_CALL tobii_set_face_type( tobii_device_t* device,
    tobii_face_type_t const face_type );
/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED */
TOBII_API tobii_error_t TOBII_CALL tobii_get_face_type( tobii_device_t* device,
    tobii_face_type_t* face_type );

#ifdef __cplusplus
}
#endif

#endif /* TOBII_ADVANCED_H */
