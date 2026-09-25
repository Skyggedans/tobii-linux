/* tobii/tobii_internal.h — the exports of tobii_stream_engine.dll 4.1.0.3
 * that the Stream Engine never documented, as provided by libtobii.so.
 *
 * Argument counts are those the DLL's code reads (tools/abi/dll_abi.py);
 * argument types are best guesses. Only the field-of-use, image,
 * internal-stream, timesync, stream-type, pause and hardware-configuration
 * functions are implemented; every other entry point here returns
 * TOBII_ERROR_NOT_SUPPORTED without reading its arguments, so the guessed
 * types cannot matter at runtime.
 * Companion to tobii/tobii.h.
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
 * valid only during the callback. timestamp_us is when the frame was taken,
 * on the tobii_system_clock clock, as every sample's (tobii_streams.h).
 * Layout from the DLL's image dispatcher. */
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

/* One tracker/host clock pair: the tracker clock (the clock of gaze data's
 * timestamp_tracker_us) read tracker_us at some host time between
 * system_start_us and system_end_us, all in microseconds. The host clock is
 * tobii_system_clock's: CLOCK_MONOTONIC, not the DLL's
 * QueryPerformanceCounter, though like it monotonic with an undefined epoch.
 * The pair comes from the first gaze frame the daemon receives after the call
 * (the tracker is started if needed), and the bracket is a fixed 30 ms
 * before the daemon read it. The DLL times a round trip instead; its own
 * offset estimator skips pairs wider than 6 ms, though it still returns them
 * with TOBII_ERROR_NO_ERROR. The callbacks' timestamp_us (and gaze data's
 * timestamp_system_us) need no pair: the daemon has already put them on the
 * host clock (tobii_streams.h); gaze data's timestamp_tracker_us is the
 * tracker time a pair maps. The tracker's clock may restart when the tracker
 * re-initialises, so a pair holds until then only. Nothing is written unless
 * the call succeeds; TOBII_ERROR_NOT_AVAILABLE while the tracker is paused,
 * TOBII_ERROR_CONNECTION_FAILED when no tracker is plugged in. Layout from
 * the DLL (24 bytes); the field names follow the older public headers. */
typedef struct tobii_timesync_data_t
{
    int64_t system_start_us;
    int64_t system_end_us;
    int64_t tracker_us;
} tobii_timesync_data_t;

TOBII_API tobii_error_t TOBII_CALL tobii_timesync( tobii_device_t* device,
    tobii_timesync_data_t* timesync );

/* One entry of the tracker's stream catalogue. `type` is the Stream Engine's
 * stream type the DLL maps the tracker's stream id to (0x500 gaze -> 1,
 * 0x501 image -> 2, 0x502 -> 3, 0x503 -> 14, 0x504 presence -> 4, 0x505 -> 5,
 * 0x506 -> 8, 0x507 -> 9, 0x508 image_collection -> 11, 0x50a -> 6,
 * 0x1770 algodbg -> 7, anything else -> 0). `value` is the tracker's number
 * for the stream (1000 for image_collection on the ET5, else 0) and `text` a
 * second string (empty on the ET5); what they mean is unknown. Offsets from
 * the DLL; the size (136 bytes) is inferred and the field names are ours. */
typedef struct tobii_stream_type_t
{
    int type;
    uint32_t value;
    char name[ 64 ];
    char text[ 64 ];
} tobii_stream_type_t;

typedef void ( *tobii_stream_type_receiver_t )( tobii_stream_type_t const* stream_type,
    void* user_data );

/* The tracker's stream catalogue, one receiver call per stream in the
 * tracker's order (nine on the ET5). It is the catalogue the tracker reported
 * at its last init, where the DLL asks the tracker on every call. It lists
 * streams this library does not deliver, such as image_collection. The DLL
 * answers only on its TTP path and with the internal feature group; this
 * library has no licence gate. TOBII_ERROR_NOT_SUPPORTED if the tracker
 * reported no catalogue (the DLL answers TOBII_ERROR_NO_ERROR with no calls)
 * or the daemon is older than this library; TOBII_ERROR_TIMED_OUT if no
 * tracker has been seen. The receiver may call back into the library; each
 * entry is valid during its call only. */
TOBII_API tobii_error_t TOBII_CALL tobii_enumerate_stream_types( tobii_device_t* device,
    tobii_stream_type_receiver_t receiver, void* user_data );

/* Pause the tracker (it sends no data until resumed), or resume it. The pause
 * is one state for the tracker, shared by every client as in the DLL: the
 * last call wins and any client may resume. Unlike the DLL, a pause or resume
 * the tracker accepts shows at once in TOBII_STATE_DEVICE_PAUSED and a
 * TOBII_NOTIFICATION_TYPE_DEVICE_PAUSED_STATE_CHANGED notification. A pause
 * ends when the client that paused last disconnects and whenever the tracker
 * re-initialises. Pausing during a calibration session is
 * TOBII_ERROR_CALIBRATION_BUSY, and no session starts while paused
 * (TOBII_ERROR_NOT_AVAILABLE). A resume the tracker does not answer still
 * succeeds: the daemon re-opens a tracker that stays silent, which resumes
 * it. */
TOBII_API tobii_error_t TOBII_CALL tobii_pause_device( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_resume_device( tobii_device_t* device );

/* PROVISIONAL. The tracker's hardware configuration (its command 2120). The
 * layout is the DLL's (2472 bytes; offsets from its copy at 0x18014abe0 and
 * its PRP deserialiser at 0x180045056, which load the 64-bit fields as
 * double); the field names are ours, and what the fields mean is not known.
 * The values come from the one answer ever captured (Windows): 16.16 values
 * unscaled, 32.32 values scaled as lengths, in mm. On Linux the ET5 has so
 * far answered 2120 with no data, so the call returns
 * TOBII_ERROR_NOT_SUPPORTED there. When it succeeds the whole struct is
 * written, zero past each count (the DLL leaves those slots untouched), and
 * a mode outside 0..2 is 0, as in the DLL. Nothing is written on an error. */
typedef struct tobii_hardware_configuration_entry_t
{
    int id;
    float param_a;
    float param_b;
    double position_xyz[ 3 ];
    double values[ 15 ];
    int width;
    int height;
    int param_c;
    int coefficient_count;
    double coefficients[ 64 ];
    double point_a_xyz[ 3 ];
    double point_b_xyz[ 3 ];
    double param_d;
} tobii_hardware_configuration_entry_t;

typedef struct tobii_hardware_configuration_t
{
    int entry_count;
    tobii_hardware_configuration_entry_t entries[ 2 ];
    int point_count;
    double points_xyz[ 40 ][ 3 ];
    int mode;
} tobii_hardware_configuration_t;

TOBII_API tobii_error_t TOBII_CALL tobii_hardware_configuration_get( tobii_device_t* device,
    tobii_hardware_configuration_t* configuration );

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
TOBII_API tobii_error_t TOBII_CALL tobii_power_save_activate( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_power_save_deactivate( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_remote_wake_activate( tobii_device_t* device );
TOBII_API tobii_error_t TOBII_CALL tobii_remote_wake_deactivate( tobii_device_t* device );
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
