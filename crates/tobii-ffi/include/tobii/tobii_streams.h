/* tobii/tobii_streams.h — the sample streams, as provided by libtobii.so.
 *
 * Companion to tobii/tobii.h; see that file for what these headers are.
 * Every sample carries the device clock in timestamp_us. Callbacks run on the
 * thread that calls tobii_device_process_callbacks; calling an implemented
 * device function from one (creating or destroying a device included),
 * tobii_calibration_parse or tobii_api_destroy returns
 * TOBII_ERROR_CALLBACK_IN_PROGRESS.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef TOBII_STREAMS_H
#define TOBII_STREAMS_H

#include "tobii.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum tobii_validity_t
{
    TOBII_VALIDITY_INVALID = 0,
    TOBII_VALIDITY_VALID = 1,
} tobii_validity_t;

/* Normalised display coordinates, the device's filtered combined gaze,
 * unclamped (looks off-screen fall outside 0..1). */
typedef struct tobii_gaze_point_t
{
    int64_t timestamp_us;
    tobii_validity_t validity;
    float position_xy[ 2 ];
} tobii_gaze_point_t;

typedef void ( *tobii_gaze_point_callback_t )( tobii_gaze_point_t const* gaze_point, void* user_data );

TOBII_API tobii_error_t TOBII_CALL tobii_gaze_point_subscribe( tobii_device_t* device,
    tobii_gaze_point_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_gaze_point_unsubscribe( tobii_device_t* device );

/* Each eye's gaze origin in the display frame (origin at the screen centre,
 * z towards the user), mm. */
typedef struct tobii_gaze_origin_t
{
    int64_t timestamp_us;
    tobii_validity_t left_validity;
    float left_xyz[ 3 ];
    tobii_validity_t right_validity;
    float right_xyz[ 3 ];
} tobii_gaze_origin_t;

typedef void ( *tobii_gaze_origin_callback_t )( tobii_gaze_origin_t const* gaze_origin, void* user_data );

TOBII_API tobii_error_t TOBII_CALL tobii_gaze_origin_subscribe( tobii_device_t* device,
    tobii_gaze_origin_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_gaze_origin_unsubscribe( tobii_device_t* device );

/* Each eye's position normalised to the track box, 0..1 per axis. */
typedef struct tobii_eye_position_normalized_t
{
    int64_t timestamp_us;
    tobii_validity_t left_validity;
    float left_xyz[ 3 ];
    tobii_validity_t right_validity;
    float right_xyz[ 3 ];
} tobii_eye_position_normalized_t;

typedef void ( *tobii_eye_position_normalized_callback_t )(
    tobii_eye_position_normalized_t const* eye_position, void* user_data );

TOBII_API tobii_error_t TOBII_CALL tobii_eye_position_normalized_subscribe(
    tobii_device_t* device, tobii_eye_position_normalized_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_eye_position_normalized_unsubscribe(
    tobii_device_t* device );

typedef enum tobii_user_presence_status_t
{
    TOBII_USER_PRESENCE_STATUS_UNKNOWN = 0,
    TOBII_USER_PRESENCE_STATUS_AWAY = 1,
    TOBII_USER_PRESENCE_STATUS_PRESENT = 2,
} tobii_user_presence_status_t;

typedef void ( *tobii_user_presence_callback_t )( tobii_user_presence_status_t status,
    int64_t timestamp_us, void* user_data );

/* Reported once on subscribe (the last known state) and then on change. */
TOBII_API tobii_error_t TOBII_CALL tobii_user_presence_subscribe( tobii_device_t* device,
    tobii_user_presence_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_user_presence_unsubscribe( tobii_device_t* device );

/* From the tracker's IR camera. Position in mm, rotation in radians about x
 * (pitch), y (yaw) and z (roll), tracker coordinates: +x right, +y up, +z
 * towards the user. */
typedef struct tobii_head_pose_t
{
    int64_t timestamp_us;
    tobii_validity_t position_validity;
    float position_xyz[ 3 ];
    tobii_validity_t rotation_validity_xyz[ 3 ];
    float rotation_xyz[ 3 ];
} tobii_head_pose_t;

typedef void ( *tobii_head_pose_callback_t )( tobii_head_pose_t const* head_pose, void* user_data );

TOBII_API tobii_error_t TOBII_CALL tobii_head_pose_subscribe( tobii_device_t* device,
    tobii_head_pose_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_head_pose_unsubscribe( tobii_device_t* device );

typedef enum tobii_notification_type_t
{
    TOBII_NOTIFICATION_TYPE_CALIBRATION_STATE_CHANGED,
    TOBII_NOTIFICATION_TYPE_EXCLUSIVE_MODE_STATE_CHANGED,
    TOBII_NOTIFICATION_TYPE_TRACK_BOX_CHANGED,
    TOBII_NOTIFICATION_TYPE_DISPLAY_AREA_CHANGED,
    TOBII_NOTIFICATION_TYPE_FRAMERATE_CHANGED,
    TOBII_NOTIFICATION_TYPE_POWER_SAVE_STATE_CHANGED,
    TOBII_NOTIFICATION_TYPE_DEVICE_PAUSED_STATE_CHANGED,
    TOBII_NOTIFICATION_TYPE_CALIBRATION_ENABLED_EYE_CHANGED,
    TOBII_NOTIFICATION_TYPE_CALIBRATION_ID_CHANGED,
    TOBII_NOTIFICATION_TYPE_COMBINED_GAZE_EYE_SELECTION_CHANGED,
    TOBII_NOTIFICATION_TYPE_FAULTS_CHANGED,
    TOBII_NOTIFICATION_TYPE_WARNINGS_CHANGED,
    TOBII_NOTIFICATION_TYPE_FACE_TYPE_CHANGED,
} tobii_notification_type_t;

typedef enum tobii_notification_value_type_t
{
    TOBII_NOTIFICATION_VALUE_TYPE_NONE,
    TOBII_NOTIFICATION_VALUE_TYPE_FLOAT,
    TOBII_NOTIFICATION_VALUE_TYPE_STATE,
    TOBII_NOTIFICATION_VALUE_TYPE_DISPLAY_AREA,
    TOBII_NOTIFICATION_VALUE_TYPE_UINT,
    TOBII_NOTIFICATION_VALUE_TYPE_ENABLED_EYE,
    TOBII_NOTIFICATION_VALUE_TYPE_STRING,
} tobii_notification_value_type_t;

/* 520 bytes. Delivered: CALIBRATION_STATE_CHANGED, DISPLAY_AREA_CHANGED,
 * DEVICE_PAUSED_STATE_CHANGED and CALIBRATION_ID_CHANGED. The enabled-eye
 * value is a tobii_enabled_eye_t (tobii_config.h), carried here as its
 * underlying int. */
typedef struct tobii_notification_t
{
    tobii_notification_type_t type;
    tobii_notification_value_type_t value_type;
    union
    {
        float float_;
        tobii_state_bool_t state;
        tobii_display_area_t display_area;
        uint32_t uint_;
        int enabled_eye;
        tobii_state_string_t string_;
    } value;
} tobii_notification_t;

typedef void ( *tobii_notifications_callback_t )( tobii_notification_t const* notification,
    void* user_data );

TOBII_API tobii_error_t TOBII_CALL tobii_notifications_subscribe( tobii_device_t* device,
    tobii_notifications_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_notifications_unsubscribe( tobii_device_t* device );

/* The track-box-normalised eye positions, as a positioning guide. */
typedef struct tobii_user_position_guide_t
{
    int64_t timestamp_us;
    tobii_validity_t left_position_validity;
    float left_position_normalized_xyz[ 3 ];
    tobii_validity_t right_position_validity;
    float right_position_normalized_xyz[ 3 ];
} tobii_user_position_guide_t;

typedef void ( *tobii_user_position_guide_callback_t )(
    tobii_user_position_guide_t const* user_position_guide, void* user_data );

TOBII_API tobii_error_t TOBII_CALL tobii_user_position_guide_subscribe( tobii_device_t* device,
    tobii_user_position_guide_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_user_position_guide_unsubscribe( tobii_device_t* device );

/* Extension, not part of the Stream Engine: make the current head pose the
 * rest pose. Affects whichever client currently holds the device's mode. */
TOBII_API tobii_error_t TOBII_CALL tobii_recenter( tobii_device_t* device );

#ifdef __cplusplus
}
#endif

#endif /* TOBII_STREAMS_H */
