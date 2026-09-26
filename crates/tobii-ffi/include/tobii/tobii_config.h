/* tobii/tobii_config.h — calibration and device configuration, as provided
 * by libtobii.so. Companion to tobii/tobii.h.
 *
 * Calibration runs in the tobiid daemon, one client at a time: another
 * client's session makes tobii_calibration_start return
 * TOBII_ERROR_CALIBRATION_BUSY. Only tobii_calibration_stop commits a
 * session: the last calibration it computed is saved as the user's
 * calibration ($XDG_CONFIG_HOME/tobii/calibration.bin) and uploaded at every
 * later init. A session that ends any other way (its owner disconnects, the
 * tracker re-initialises or goes away) saves nothing: the calibration and
 * display area it started from go back, and its owner's later calls in it,
 * tobii_calibration_stop included, return TOBII_ERROR_CALIBRATION_NOT_STARTED.
 * Only 2-D calibration of both eyes is supported.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef TOBII_CONFIG_H
#define TOBII_CONFIG_H

#include "tobii.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum tobii_enabled_eye_t
{
    TOBII_ENABLED_EYE_LEFT,
    TOBII_ENABLED_EYE_RIGHT,
    TOBII_ENABLED_EYE_BOTH,
} tobii_enabled_eye_t;

/* Only TOBII_ENABLED_EYE_BOTH. */
TOBII_API tobii_error_t TOBII_CALL tobii_set_enabled_eye( tobii_device_t* device,
    tobii_enabled_eye_t enabled_eye );
TOBII_API tobii_error_t TOBII_CALL tobii_get_enabled_eye( tobii_device_t* device,
    tobii_enabled_eye_t* enabled_eye );

/* Only TOBII_ENABLED_EYE_BOTH. */
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_start( tobii_device_t* device,
    tobii_enabled_eye_t enabled_eye );
/* Keeps the last calibration computed and the display area set in the
 * session. Restores the previous ones only when nothing was computed, or when
 * the daemon cannot save the calibration (TOBII_ERROR_OPERATION_FAILED). Once
 * saved, both are kept even if the tracker then refuses the calibration or
 * goes away before taking it (TOBII_ERROR_OPERATION_FAILED or
 * TOBII_ERROR_CONNECTION_FAILED all the same): it loads them at its next
 * init. */
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_stop( tobii_device_t* device );
/* Normalised display coordinates; blocks for most of a second. */
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_collect_data_2d( tobii_device_t* device,
    float x, float y );
/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED */
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_collect_data_3d( tobii_device_t* device,
    float x, float y, float z );
/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED */
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_collect_data_per_eye_2d( tobii_device_t* device,
    float x, float y, tobii_enabled_eye_t requested_eyes, tobii_enabled_eye_t* collected_eyes );
/* Discards the data collected at (x, y) in this session; pass the point as collected. */
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_discard_data_2d( tobii_device_t* device,
    float x, float y );
/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED */
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_discard_data_3d( tobii_device_t* device,
    float x, float y, float z );
/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED */
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_discard_data_per_eye_2d( tobii_device_t* device,
    float x, float y, tobii_enabled_eye_t eyes );
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_clear( tobii_device_t* device );
/* Active at once, but saved only by tobii_calibration_stop. */
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_compute_and_apply( tobii_device_t* device );
/* NOT IMPLEMENTED: returns TOBII_ERROR_NOT_SUPPORTED */
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_compute_and_apply_per_eye(
    tobii_device_t* device, tobii_enabled_eye_t* calibrated_eyes );
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_retrieve( tobii_device_t* device,
    tobii_data_receiver_t receiver, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_apply( tobii_device_t* device,
    void const* data, size_t size );

typedef enum tobii_calibration_point_status_t
{
    TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID,
    TOBII_CALIBRATION_POINT_STATUS_VALID_BUT_NOT_USED_IN_CALIBRATION,
    TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION,
} tobii_calibration_point_status_t;

typedef struct tobii_calibration_point_data_t
{
    float point_xy[ 2 ];
    tobii_calibration_point_status_t left_status;
    float left_mapping_xy[ 2 ];
    tobii_calibration_point_status_t right_status;
    float right_mapping_xy[ 2 ];
} tobii_calibration_point_data_t;

typedef void ( *tobii_calibration_point_data_receiver_t )(
    tobii_calibration_point_data_t const* point_data, void* user_data );

/* The ET5 keeps 14 points: two rounds of the 7-point pattern. Data that is not a
 * valid calibration is TOBII_ERROR_OPERATION_FAILED, before any point is passed.
 * The DLL refuses only a negative point count and parses everything else; this
 * library also refuses a short blob, bytes after the point list, a status word
 * other than -1, 0, 1 or 2 (its upper 32 bits 0, or all ones with -1), and a
 * value that is not finite or lies outside the display by more than half its
 * size, unless it is the mapping of an eye marked failed (-1). */
TOBII_API tobii_error_t TOBII_CALL tobii_calibration_parse( tobii_api_t* api, void const* data,
    size_t data_size, tobii_calibration_point_data_receiver_t receiver, void* user_data );

typedef struct tobii_geometry_mounting_t
{
    int guides;
    float width_mm;
    float angle_deg;
    float external_offset_mm_xyz[ 3 ];
    float internal_offset_mm_xyz[ 3 ];
} tobii_geometry_mounting_t;

TOBII_API tobii_error_t TOBII_CALL tobii_get_geometry_mounting( tobii_device_t* device,
    tobii_geometry_mounting_t* geometry_mounting );
TOBII_API tobii_error_t TOBII_CALL tobii_get_display_area( tobii_device_t* device,
    tobii_display_area_t* display_area );
/* Kept by the daemon and re-applied at every device re-init. */
TOBII_API tobii_error_t TOBII_CALL tobii_set_display_area( tobii_device_t* device,
    tobii_display_area_t const* display_area );
TOBII_API tobii_error_t TOBII_CALL tobii_calculate_display_area_basic( tobii_api_t* api,
    float width_mm, float height_mm, float offset_x_mm,
    tobii_geometry_mounting_t const* geometry_mounting, tobii_display_area_t* display_area );

typedef char tobii_device_name_t[ 64 ];

/* The name a client set, else the device's model string. Asked of the daemon on
 * every call: another process may have renamed the device. */
TOBII_API tobii_error_t TOBII_CALL tobii_get_device_name( tobii_device_t* device,
    tobii_device_name_t* device_name );
/* Kept by the daemon ($XDG_CONFIG_HOME/tobii/device-name) for every client and
 * later sessions; nothing is written to the tracker. At most 63 bytes are read,
 * up to the NUL. A NULL name is TOBII_ERROR_INVALID_PARAMETER. */
TOBII_API tobii_error_t TOBII_CALL tobii_set_device_name( tobii_device_t* device,
    tobii_device_name_t const device_name );

typedef void ( *tobii_output_frequency_receiver_t )( float output_frequency, void* user_data );

/* The ET5 runs at 33 Hz; other frequencies are TOBII_ERROR_NOT_SUPPORTED. */
TOBII_API tobii_error_t TOBII_CALL tobii_enumerate_output_frequencies( tobii_device_t* device,
    tobii_output_frequency_receiver_t receiver, void* user_data );
TOBII_API tobii_error_t TOBII_CALL tobii_set_output_frequency( tobii_device_t* device,
    float output_frequency );
TOBII_API tobii_error_t TOBII_CALL tobii_get_output_frequency( tobii_device_t* device,
    float* output_frequency );

#ifdef __cplusplus
}
#endif

#endif /* TOBII_CONFIG_H */
