/* tobii/tobii_internal.h — the exports of tobii_stream_engine.dll 4.1.0.3
 * that the Stream Engine never documented, as provided by libtobii.so.
 *
 * Argument counts are those the DLL's code reads (tools/abi/dll_abi.py);
 * argument types are best guesses. Only the field-of-use, image,
 * internal-stream and timesync functions are implemented; every other entry
 * point here returns TOBII_ERROR_NOT_SUPPORTED without reading its arguments,
 * so the guessed types cannot matter at runtime. Companion to tobii/tobii.h.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef TOBII_INTERNAL_H
#define TOBII_INTERNAL_H

#include "tobii.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void ( *tobii_field_of_use_callback_t )( tobii_field_of_use_t field_of_use, void* user_data );

/* The field of use the device was created with. */
TOBII_API tobii_error_t TOBII_CALL tobii_get_field_of_use( tobii_device_t* device,
    tobii_field_of_use_t* field_of_use );
/* Accepted; the callback is never called (the field of use cannot change). */
TOBII_API tobii_error_t TOBII_CALL tobii_field_of_use_subscribe( tobii_device_t* device,
    tobii_field_of_use_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_field_of_use_unsubscribe( tobii_device_t* device );

/* An IR camera frame: 280x280, 8 bits per pixel, ~33 Hz on the ET5. `data` is
 * valid only during the callback. Layout from the DLL's image dispatcher. */
typedef struct tobii_image_t
{
    int64_t timestamp_us;
    int width;
    int padding_per_row;
    int height;
    int bits_per_pixel;
    void const* data;
} tobii_image_t;

typedef void ( *tobii_image_callback_t )( tobii_image_t const* image, void* user_data );

TOBII_API tobii_error_t TOBII_CALL tobii_image_subscribe( tobii_device_t* device,
    tobii_image_callback_t callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_image_unsubscribe( tobii_device_t* device );

/* Internal stream ids (the names of 0..2 are inferred): 0 image, 1 clean IR,
 * 2 custom, 3 low-frequency head rotation, 4 low-frequency head position,
 * 5 multiple faces position, 6 image collection, 7 wearable limited image,
 * 8 secondary camera image. Supported: 0 (the IR image) only, which is what
 * this library delivers; it is not the DLL's answer. An unknown id is
 * reported unsupported, not an error; an id above INT32_MAX is
 * TOBII_ERROR_INVALID_PARAMETER. */
TOBII_API tobii_error_t TOBII_CALL tobii_internal_stream_supported( tobii_device_t* device,
    uint32_t stream, tobii_supported_t* supported );

/* One tracker/host clock pair: the tracker clock (the clock of the samples'
 * timestamps) read tracker_us at some host time between system_start_us and
 * system_end_us, all in microseconds. The host clock is tobii_system_clock's:
 * CLOCK_REALTIME, not the DLL's QueryPerformanceCounter. The pair comes from
 * the first gaze frame the daemon receives after the call (the tracker is
 * started if needed), and the bracket is a fixed 30 ms before the daemon
 * read it. The DLL times a round trip instead; its own offset estimator
 * skips pairs wider than 6 ms, though it still returns them with
 * TOBII_ERROR_NO_ERROR. Nothing is written unless the call succeeds. Layout
 * from the DLL (24 bytes); the field names follow the older public headers. */
typedef struct tobii_timesync_data_t
{
    int64_t system_start_us;
    int64_t system_end_us;
    int64_t tracker_us;
} tobii_timesync_data_t;

TOBII_API tobii_error_t TOBII_CALL tobii_timesync( tobii_device_t* device,
    tobii_timesync_data_t* timesync );

/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED. Its point type is
 * unknown. */
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_stimulus_points_get( tobii_device_t* device,
    void* points );

/* Everything below: NOT IMPLEMENTED, returns TOBII_ERROR_NOT_SUPPORTED. */

TOBII_API tobii_error_t TOBII_CALL tobii_clean_ir_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_clean_ir_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_custom_stream_subscribe( tobii_device_t* device, void const* callback, uint32_t stream, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_custom_stream_unsubscribe( tobii_device_t* device, uint32_t stream );
TOBII_API tobii_error_t TOBII_CALL tobii_diagnostic_images_retrieve( tobii_device_t* device, void const* receiver, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_diagnostics_dump_images( tobii_device_t* device, uint32_t a, uint32_t b );
TOBII_API tobii_error_t TOBII_CALL tobii_diagnostics_get_data( tobii_device_t* device, uint32_t kind, void const* receiver, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_diagnostics_image_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_diagnostics_image_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_display_id_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_display_id_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_enable_extension( tobii_device_t* device, uint32_t extension );
TOBII_API tobii_error_t TOBII_CALL tobii_enumerate_enabled_extensions( tobii_device_t* device, void const* receiver, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_enumerate_extensions( tobii_device_t* device, void const* receiver, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_enumerate_illumination_modes( tobii_device_t* device, void const* receiver, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_enumerate_stream_type_columns( tobii_device_t* device, uint32_t stream_type, void const* receiver, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_enumerate_stream_types( tobii_device_t* device, void const* receiver, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_face_id_enroll( tobii_device_t* device, void* a, void* b );
TOBII_API tobii_error_t TOBII_CALL tobii_face_id_enroll_clear( tobii_device_t* device, void* a );
TOBII_API tobii_error_t TOBII_CALL tobii_face_id_parameters_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_face_id_parameters_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_face_id_state_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_face_id_state_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_foveated_rendering_gaze_point_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_foveated_rendering_gaze_point_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_gaze_raw_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_gaze_raw_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_get_combined_gaze_hid_track_box( tobii_device_t* device, void* track_box );
TOBII_API tobii_error_t TOBII_CALL tobii_get_configuration_key( tobii_device_t* device, void* key, void* value );
TOBII_API tobii_error_t TOBII_CALL tobii_get_device_info_internal( tobii_device_t* device, void* info );
TOBII_API tobii_error_t TOBII_CALL tobii_get_display_id( tobii_device_t* device, void* display_id );
TOBII_API tobii_error_t TOBII_CALL tobii_get_display_info( tobii_device_t* device, void* display_info );
TOBII_API tobii_error_t TOBII_CALL tobii_get_face_id_parameters( tobii_device_t* device, void* parameters );
TOBII_API tobii_error_t TOBII_CALL tobii_get_face_id_state( tobii_device_t* device, void* state );
TOBII_API tobii_error_t TOBII_CALL tobii_get_gaze_hid_enabled( tobii_device_t* device, void* enabled );
TOBII_API tobii_error_t TOBII_CALL tobii_get_illumination_mode( tobii_device_t* device, void* mode );
TOBII_API tobii_error_t TOBII_CALL tobii_hardware_configuration_get( tobii_device_t* device, void* configuration );
TOBII_API tobii_error_t TOBII_CALL tobii_image_collection_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_image_collection_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_internal_capability_supported( tobii_device_t* device, uint32_t capability, void* supported );
TOBII_API tobii_error_t TOBII_CALL tobii_logs_retrieve( tobii_device_t* device, void const* receiver, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_low_frequency_head_position_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_low_frequency_head_position_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_low_frequency_head_rotation_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_low_frequency_head_rotation_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_multiple_faces_position_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_multiple_faces_position_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_open_realm( tobii_device_t* device, uint32_t realm, void const* key, uint32_t key_size );
TOBII_API tobii_error_t TOBII_CALL tobii_pause_device( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_power_save_activate( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_power_save_deactivate( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_remote_wake_activate( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_remote_wake_deactivate( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_resume_device( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_secondary_camera_image_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_secondary_camera_image_unsubscribe( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_send_custom_command( tobii_device_t* device, uint32_t command, void const* data, size_t size, void const* receiver, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_send_statistics( tobii_device_t* device, void const* data, size_t size );
TOBII_API tobii_error_t TOBII_CALL tobii_set_display_id( tobii_device_t* device, uint32_t display_id );
TOBII_API tobii_error_t TOBII_CALL tobii_set_display_info( tobii_device_t* device, void const* display_info );
TOBII_API tobii_error_t TOBII_CALL tobii_set_face_id_parameters( tobii_device_t* device, void const* parameters );
TOBII_API tobii_error_t TOBII_CALL tobii_set_fw_upgrade_allowed( tobii_device_t* device, uint32_t allowed );
TOBII_API tobii_error_t TOBII_CALL tobii_set_illumination_mode( tobii_device_t* device, void const* mode );
TOBII_API tobii_error_t TOBII_CALL tobii_wearable_limited_image_subscribe( tobii_device_t* device, void const* callback, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_wearable_limited_image_unsubscribe( tobii_device_t* device );

#ifdef __cplusplus
}
#endif

#endif /* TOBII_INTERNAL_H */
