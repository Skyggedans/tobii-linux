/* tobii/tobii.h — the core of the Tobii Stream Engine 4.1 C API, as provided
 * by libtobii.so.
 *
 * Not Tobii's header: an independent declaration of the same interface, so
 * code written against the Stream Engine builds and links unchanged on Linux.
 * Signatures, enum values and struct layouts are those of the reference
 * tobii_stream_engine.dll 4.1.0.3 (see tools/abi/README.md for how each was
 * established). libtobii.so exports all 153 of its entry points; the ones
 * marked NOT IMPLEMENTED return TOBII_ERROR_NOT_SUPPORTED.
 *
 * The companion headers follow the Stream Engine's split: tobii_streams.h,
 * tobii_config.h, tobii_licensing.h, tobii_advanced.h, tobii_wearable.h, and
 * tobii_internal.h for the exports the Stream Engine never documented.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef TOBII_TOBII_H
#define TOBII_TOBII_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TOBII_CALL
#define TOBII_API

typedef enum tobii_error_t
{
    TOBII_ERROR_NO_ERROR,                       /*  0 */
    TOBII_ERROR_INTERNAL,                       /*  1 */
    TOBII_ERROR_INSUFFICIENT_LICENSE,           /*  2 */
    TOBII_ERROR_NOT_SUPPORTED,                  /*  3 */
    TOBII_ERROR_NOT_AVAILABLE,                  /*  4 */
    TOBII_ERROR_CONNECTION_FAILED,              /*  5 */
    TOBII_ERROR_TIMED_OUT,                      /*  6 */
    TOBII_ERROR_ALLOCATION_FAILED,              /*  7 */
    TOBII_ERROR_INVALID_PARAMETER,              /*  8 */
    TOBII_ERROR_CALIBRATION_ALREADY_STARTED,    /*  9 */
    TOBII_ERROR_CALIBRATION_NOT_STARTED,        /* 10 */
    TOBII_ERROR_ALREADY_SUBSCRIBED,             /* 11 */
    TOBII_ERROR_NOT_SUBSCRIBED,                 /* 12 */
    TOBII_ERROR_OPERATION_FAILED,               /* 13 */
    TOBII_ERROR_CONFLICTING_API_INSTANCES,      /* 14 */
    TOBII_ERROR_CALIBRATION_BUSY,               /* 15 */
    TOBII_ERROR_CALLBACK_IN_PROGRESS,           /* 16 */
    TOBII_ERROR_TOO_MANY_SUBSCRIBERS,           /* 17 */
    TOBII_ERROR_CONNECTION_FAILED_DRIVER,       /* 18 */
    TOBII_ERROR_UNAUTHORIZED,                   /* 19 */
    TOBII_ERROR_FIRMWARE_UPGRADE_IN_PROGRESS,   /* 20 */
} tobii_error_t;

typedef struct tobii_version_t
{
    int major;
    int minor;
    int revision;
    int build;
} tobii_version_t;

typedef struct tobii_api_t tobii_api_t;
typedef struct tobii_device_t tobii_device_t;

/* Accepted by tobii_api_create for signature compatibility, then ignored:
 * libtobii.so allocates with Rust's allocator and logs through `tracing`
 * (RUST_LOG). Pass NULL for both. */
typedef void* ( *tobii_malloc_func_t )( void* mem_context, size_t size );
typedef void ( *tobii_free_func_t )( void* mem_context, void* ptr );

typedef struct tobii_custom_alloc_t
{
    void* mem_context;
    tobii_malloc_func_t malloc_func;
    tobii_free_func_t free_func;
} tobii_custom_alloc_t;

typedef enum tobii_log_level_t
{
    TOBII_LOG_LEVEL_ERROR,
    TOBII_LOG_LEVEL_WARN,
    TOBII_LOG_LEVEL_INFO,
    TOBII_LOG_LEVEL_DEBUG,
    TOBII_LOG_LEVEL_TRACE,
} tobii_log_level_t;

typedef void ( *tobii_log_func_t )( void* log_context, tobii_log_level_t level, char const* text );

typedef struct tobii_custom_log_t
{
    void* log_context;
    tobii_log_func_t log_func;
} tobii_custom_log_t;

typedef void ( *tobii_device_url_receiver_t )( char const* url, void* user_data );
typedef void ( *tobii_data_receiver_t )( void const* data, size_t size, void* user_data );

#define TOBII_DEVICE_GENERATION_G5  0x00000002
#define TOBII_DEVICE_GENERATION_IS3 0x00000004
#define TOBII_DEVICE_GENERATION_IS4 0x00000008

/* Rejected unless 1 or 2 — which is also what catches a caller built against
 * the pre-4.0 header, whose three-argument tobii_device_create would
 * otherwise land its device pointer in this slot. */
typedef enum tobii_field_of_use_t
{
    TOBII_FIELD_OF_USE_INTERACTIVE = 1,
    TOBII_FIELD_OF_USE_ANALYTICAL = 2,
} tobii_field_of_use_t;

typedef struct tobii_device_info_t
{
    char serial_number[ 256 ];
    char model[ 256 ];
    char generation[ 256 ];
    char firmware_version[ 256 ];
    char integration_id[ 128 ];
    char hw_calibration_version[ 128 ];
    char hw_calibration_date[ 128 ];
    char lot_id[ 128 ];
    char integration_type[ 256 ];
    char runtime_build_version[ 256 ];
} tobii_device_info_t;

typedef struct tobii_track_box_t
{
    float front_upper_right_xyz[ 3 ];
    float front_upper_left_xyz[ 3 ];
    float front_lower_left_xyz[ 3 ];
    float front_lower_right_xyz[ 3 ];
    float back_upper_right_xyz[ 3 ];
    float back_upper_left_xyz[ 3 ];
    float back_lower_left_xyz[ 3 ];
    float back_lower_right_xyz[ 3 ];
} tobii_track_box_t;

/* Shared by tobii_streams.h (notifications) and tobii_config.h. Tracker
 * frame, mm. */
typedef struct tobii_display_area_t
{
    float top_left_mm_xyz[ 3 ];
    float top_right_mm_xyz[ 3 ];
    float bottom_left_mm_xyz[ 3 ];
} tobii_display_area_t;

typedef enum tobii_state_t
{
    TOBII_STATE_POWER_SAVE_ACTIVE,
    TOBII_STATE_REMOTE_WAKE_ACTIVE,
    TOBII_STATE_DEVICE_PAUSED,
    TOBII_STATE_EXCLUSIVE_MODE,
    TOBII_STATE_FAULT,
    TOBII_STATE_WARNING,
    TOBII_STATE_CALIBRATION_ID,
    TOBII_STATE_CALIBRATION_ACTIVE,
} tobii_state_t;

typedef enum tobii_state_bool_t
{
    TOBII_STATE_BOOL_FALSE,
    TOBII_STATE_BOOL_TRUE,
} tobii_state_bool_t;

typedef char tobii_state_string_t[ 512 ];

typedef enum tobii_capability_t
{
    TOBII_CAPABILITY_DISPLAY_AREA_WRITABLE,
    TOBII_CAPABILITY_CALIBRATION_2D,
    TOBII_CAPABILITY_CALIBRATION_3D,
    TOBII_CAPABILITY_PERSISTENT_STORAGE,
    TOBII_CAPABILITY_CALIBRATION_PER_EYE,
    TOBII_CAPABILITY_COMPOUND_STREAM_WEARABLE_3D_GAZE_COMBINED,
    TOBII_CAPABILITY_FACE_TYPE,
    TOBII_CAPABILITY_COMPOUND_STREAM_USER_POSITION_GUIDE_XY,
    TOBII_CAPABILITY_COMPOUND_STREAM_USER_POSITION_GUIDE_Z,
    TOBII_CAPABILITY_COMPOUND_STREAM_WEARABLE_LIMITED_IMAGE,
    TOBII_CAPABILITY_COMPOUND_STREAM_WEARABLE_PUPIL_DIAMETER,
    TOBII_CAPABILITY_COMPOUND_STREAM_WEARABLE_PUPIL_POSITION,
    TOBII_CAPABILITY_COMPOUND_STREAM_WEARABLE_EYE_OPENNESS,
    TOBII_CAPABILITY_COMPOUND_STREAM_WEARABLE_3D_GAZE_PER_EYE,
    TOBII_CAPABILITY_COMPOUND_STREAM_WEARABLE_USER_POSITION_GUIDE_XY,
    TOBII_CAPABILITY_COMPOUND_STREAM_WEARABLE_TRACKING_IMPROVEMENTS,
    TOBII_CAPABILITY_COMPOUND_STREAM_WEARABLE_CONVERGENCE_DISTANCE,
    TOBII_CAPABILITY_COMPOUND_STREAM_WEARABLE_IMPROVE_USER_POSITION_HMD,
    TOBII_CAPABILITY_COMPOUND_STREAM_WEARABLE_INCREASE_EYE_RELIEF,
} tobii_capability_t;

typedef enum tobii_supported_t
{
    TOBII_NOT_SUPPORTED,
    TOBII_SUPPORTED,
} tobii_supported_t;

/* Numbered as tobii_stream_engine.dll 4.1.0.3 numbers them (see
 * tools/abi/README.md). The 4.1 documentation still lists the pre-4.0 names,
 * among them WEARABLE and CUSTOM, which have no value in the DLL. */
typedef enum tobii_stream_t
{
    TOBII_STREAM_GAZE_POINT = 0,
    TOBII_STREAM_GAZE_ORIGIN = 1,
    TOBII_STREAM_EYE_POSITION_NORMALIZED = 2,
    TOBII_STREAM_USER_PRESENCE = 3,
    TOBII_STREAM_HEAD_POSE = 4,
    TOBII_STREAM_GAZE_DATA = 5,
    TOBII_STREAM_DIGITAL_SYNCPORT = 6,
    TOBII_STREAM_DIAGNOSTICS_IMAGE = 7,
    TOBII_STREAM_USER_POSITION_GUIDE = 8,
    TOBII_STREAM_WEARABLE_CONSUMER = 9,
    TOBII_STREAM_WEARABLE_ADVANCED = 10,
    TOBII_STREAM_WEARABLE_FOVEATED_GAZE = 11,
} tobii_stream_t;

/* Static, never NULL, valid for the lifetime of the process. */
TOBII_API char const* TOBII_CALL tobii_error_message( tobii_error_t error );

/* 4.1.0.3: the version this library imitates. */
TOBII_API tobii_error_t TOBII_CALL tobii_get_api_version( tobii_version_t* version );

TOBII_API tobii_error_t TOBII_CALL tobii_api_create( tobii_api_t** api,
    tobii_custom_alloc_t const* custom_alloc, tobii_custom_log_t const* custom_log );
TOBII_API tobii_error_t TOBII_CALL tobii_api_destroy( tobii_api_t* api );

/* Calls `receiver` once: libtobii.so is backed by the tobiid daemon, which
 * owns exactly one device. Whether a tracker is attached surfaces at
 * tobii_device_create. */
TOBII_API tobii_error_t TOBII_CALL tobii_enumerate_local_device_urls( tobii_api_t* api,
    tobii_device_url_receiver_t receiver, void* user_data );
/* The ET5 is always enumerated for a non-zero filter; zero is invalid. */
TOBII_API tobii_error_t TOBII_CALL tobii_enumerate_local_device_urls_ex( tobii_api_t* api,
    tobii_device_url_receiver_t receiver, void* user_data, uint32_t device_generations );

/* `url` is accepted and ignored. Returns TOBII_ERROR_CONNECTION_FAILED if the
 * daemon cannot be reached (it is spawned on demand). */
TOBII_API tobii_error_t TOBII_CALL tobii_device_create( tobii_api_t* api, char const* url,
    tobii_field_of_use_t field_of_use, tobii_device_t** device );
TOBII_API tobii_error_t TOBII_CALL tobii_device_destroy( tobii_device_t* device );

/* Blocks until a device has a sample queued, or ~100 ms per idle device. */
TOBII_API tobii_error_t TOBII_CALL tobii_wait_for_callbacks( int device_count,
    tobii_device_t* const* devices );
TOBII_API tobii_error_t TOBII_CALL tobii_device_process_callbacks( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_device_clear_callback_buffers( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_device_reconnect( tobii_device_t* device );
/* A no-op: samples carry the device clock, and there is no offset estimate to
 * refresh (tobii_timesync takes a fresh clock pair on every call). */
TOBII_API tobii_error_t TOBII_CALL tobii_update_timesync( tobii_device_t* device );
/* Microseconds since the Unix epoch: the clock of gaze data's
 * timestamp_system_us. */
TOBII_API tobii_error_t TOBII_CALL tobii_system_clock( tobii_api_t* api, int64_t* timestamp_us );

/* Serial, model, generation and firmware; runtime_build_version names
 * libtobii.so; the other fields are empty. */
TOBII_API tobii_error_t TOBII_CALL tobii_get_device_info( tobii_device_t* device,
    tobii_device_info_t* device_info );
TOBII_API tobii_error_t TOBII_CALL tobii_get_track_box( tobii_device_t* device,
    tobii_track_box_t* track_box );

/* Bool states 0, 1, 3, 4 and 5 are always false; DEVICE_PAUSED and
 * CALIBRATION_ACTIVE come from the daemon. Unlike the DLL, DEVICE_PAUSED
 * changes as soon as the tracker accepts a pause or resume, and a tracker
 * re-init ends a pause. The uint32 state is CALIBRATION_ID (DEVICE_PAUSED
 * there is TOBII_ERROR_INVALID_PARAMETER, as in the DLL); the string states
 * (FAULT, WARNING) are always empty. */
TOBII_API tobii_error_t TOBII_CALL tobii_get_state_bool( tobii_device_t* device,
    tobii_state_t state, tobii_state_bool_t* value );
TOBII_API tobii_error_t TOBII_CALL tobii_get_state_uint32( tobii_device_t* device,
    tobii_state_t state, uint32_t* value );
TOBII_API tobii_error_t TOBII_CALL tobii_get_state_string( tobii_device_t* device,
    tobii_state_t state, tobii_state_string_t value );

/* Supported: DISPLAY_AREA_WRITABLE, CALIBRATION_2D, USER_POSITION_GUIDE_XY/Z.
 * An unknown value is reported unsupported, not an error. */
TOBII_API tobii_error_t TOBII_CALL tobii_capability_supported( tobii_device_t* device,
    tobii_capability_t capability, tobii_supported_t* supported );
/* Supported: GAZE_POINT, GAZE_ORIGIN, EYE_POSITION_NORMALIZED, USER_PRESENCE,
 * HEAD_POSE, GAZE_DATA, USER_POSITION_GUIDE: the streams libtobii.so
 * delivers. Any other value, including one above WEARABLE_FOVEATED_GAZE, is
 * reported unsupported, not an error; a negative value is
 * TOBII_ERROR_INVALID_PARAMETER. */
TOBII_API tobii_error_t TOBII_CALL tobii_stream_supported( tobii_device_t* device,
    tobii_stream_t stream, tobii_supported_t* supported );

#ifdef __cplusplus
}
#endif

#endif /* TOBII_TOBII_H */
