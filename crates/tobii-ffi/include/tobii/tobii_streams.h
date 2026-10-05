/* tobii/tobii_streams.h — the sample streams, as provided by libtobii.so.
 *
 * Companion to tobii/tobii.h; see that file for what these headers are.
 * Every sample's timestamp_us is when it was taken, on the clock
 * tobii_system_clock reads. The daemon maps the tracker's clock onto it by an
 * offset it estimates from the arrivals: the smallest receipt-minus-tracker
 * time of the gaze frames and IR images of the last 120 s, started afresh
 * at every tracker init (which may restart the tracker's clock). A stamp is
 * no later than the daemon's read of its sample (a microsecond past it when
 * two samples of a stream are read together; one read during the init gets
 * the time it is delivered, a few ms late). While the tracker stays open the
 * stamps of each stream strictly increase; across an init they run on with
 * the host clock instead of jumping back. A presence reported again on
 * subscribe or tobii_device_reconnect keeps the stamp it was last reported
 * with. A head pose has the time of the IR image it was made from. Tracker
 * time is left only in gaze data's timestamp_tracker_us (tobii_advanced.h),
 * raw gaze's timestamp_tracker_us and tobii_timesync's tracker_us
 * (tobii_internal.h).
 * Callbacks run on the thread that calls tobii_device_process_callbacks, one
 * at a time per device, whichever thread that is: a callback and its
 * user_data must be sound to use on any thread that processes the device.
 * Calling an implemented device function from one (creating or destroying a
 * device included), tobii_calibration_parse or tobii_api_destroy returns
 * TOBII_ERROR_CALLBACK_IN_PROGRESS, on that thread; other threads' calls go
 * on. A callback must not block on another thread's call into any device
 * (tobii.h, Threads). A subscribe and an unsubscribe each wait for a
 * callback of the device that another thread is running. Once an
 * unsubscribe returns, its callback is not running on any thread, and none
 * calls it again. A subscribe sets its callback before it asks tobiid, and
 * lets the device's callbacks run during that round trip, so its own may be
 * called, on a thread processing the device, before it returns, even if it
 * then fails; once it has returned an error, none calls it again. The DLL
 * stores a callback only once the tracker has taken the subscription, and
 * holds the device's other callbacks back meanwhile, so a failed
 * subscribe's never runs there.
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

/* Reported once on subscribe (the last known state, with the time it was
 * last reported; not while paused) and then on change. */
TOBII_API tobii_error_t TOBII_CALL tobii_user_presence_subscribe( tobii_device_t* device,
    tobii_user_presence_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_user_presence_unsubscribe( tobii_device_t* device );

/* The Stream Engine's head pose, from the tracker's IR images: one for every
 * image tobiid processes (~33 Hz), valid or not; a value whose validity is
 * TOBII_VALIDITY_INVALID holds no measurement. Absolute, in the display
 * frame: mm from the centre of the display area, +x right and +y up along
 * the display as the user sees it, +z out of it towards the user. The
 * position is a point on the camera's line of sight to the point midway
 * between the eyes, at a range set by the head's size in the image, so
 * turning the head moves it. The rotation is in radians, right-handed about
 * those axes: the yxz Euler angles of R = Ry(y) Rx(x) Rz(z), as the Stream
 * Engine's, all zero for a face square to the display; +x lifts the chin
 * (pitch), +y turns the head to the user's left (yaw), +z tilts it towards
 * the left shoulder (roll). Nothing in it is relative to a rest pose, so
 * tobii_recenter leaves it alone. A tobiid older than libtobii.so acks the
 * subscription but sends none: restart tobiid after installing (libtobii
 * logs a warning, once per device, after 20 s without one). */
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
 * DEVICE_PAUSED_STATE_CHANGED, CALIBRATION_ID_CHANGED, and FAULTS_CHANGED /
 * WARNINGS_CHANGED (STRING: the new list, cut to 511 bytes, on every one the
 * tracker announces, as the DLL). The other types are never sent for this
 * tracker. The enabled-eye value is a tobii_enabled_eye_t (tobii_config.h),
 * carried here as its underlying int. */
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

/* Extension, not part of the Stream Engine: make tobiid's current head pose
 * the rest pose of its own, relative head pose, the legacy stream its
 * tobii-opentrack bridge reads, for every client of that stream. It does
 * not touch tobii_head_pose_subscribe's pose, the Stream Engine's, which is
 * absolute and has no rest pose, as the Stream Engine has no recenter: an
 * application centres it itself, as OpenTrack does. */
TOBII_API tobii_error_t TOBII_CALL tobii_recenter( tobii_device_t* device );

#ifdef __cplusplus
}
#endif

#endif /* TOBII_STREAMS_H */
