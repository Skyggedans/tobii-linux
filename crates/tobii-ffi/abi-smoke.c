/* Proves that a Stream-Engine-shaped C program compiles against the tobii
 * headers, links against libtobii.so and gets the answers they promise.
 * Needs neither the tobiid daemon nor a tracker: every call here is answered
 * locally or rejected before any device work would happen.
 *
 * Built and run by `make verify-abi`, which generates abi-symbols.inc (one
 * X(name) per exported symbol) from crates/tobii-ffi/abi-symbols.txt.
 *
 * SPDX-License-Identifier: MIT
 */

/* clock_gettime is POSIX, not C11. */
#define _POSIX_C_SOURCE 200809L

#include <tobii/tobii.h>
#include <tobii/tobii_advanced.h>
#include <tobii/tobii_config.h>
#include <tobii/tobii_internal.h>
#include <tobii/tobii_licensing.h>
#include <tobii/tobii_streams.h>
#include <tobii/tobii_wearable.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* Every listed symbol, by address: each must be declared (with C linkage) by
 * some header, or this does not compile, and exported, or it does not link. */
typedef void ( *any_fn_t )( void );
#define X( name ) ( any_fn_t ) & name,
static any_fn_t const abi_symbols[] = {
#include "abi-symbols.inc"
};
#undef X

static int url_count = 0;
static char first_url[ 256 ] = { 0 };

static void url_receiver( char const* url, void* user_data )
{
    assert( user_data == &url_count );
    assert( url != NULL );
    if( url_count++ == 0 && strlen( url ) < sizeof( first_url ) )
        strcpy( first_url, url );
}

static void head_pose_callback( tobii_head_pose_t const* head_pose, void* user_data )
{
    (void)head_pose;
    (void)user_data;
}

/* How far the system clock reads behind the wall clock at least: 1e15 us,
 * some 31.7 years, where the wall clock is some 1.8e15 us past its epoch and
 * CLOCK_MONOTONIC counts from boot. */
static int64_t const far_from_the_wall_clock_us = 1000000000000000LL;

static int64_t clock_us( clockid_t clock )
{
    struct timespec ts = { 0 };
    int const rc = clock_gettime( clock, &ts );
    assert( rc == 0 );
    (void)rc;
    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

int main( void )
{
    size_t const exported = sizeof( abi_symbols ) / sizeof( abi_symbols[ 0 ] );
    assert( exported == 154 );
    for( size_t i = 0; i < exported; ++i )
        assert( abi_symbols[ i ] != NULL );

    /* The callback types must accept the plain functions clients write. */
    tobii_head_pose_callback_t const head_pose_cb = head_pose_callback;
    assert( head_pose_cb != NULL );

    tobii_version_t version;
    assert( tobii_get_api_version( &version ) == TOBII_ERROR_NO_ERROR );
    assert( version.major == 4 && version.minor == 1 && version.revision == 0 && version.build == 3 );

    tobii_api_t* api = NULL;
    assert( tobii_api_create( &api, NULL, NULL ) == TOBII_ERROR_NO_ERROR );
    assert( api != NULL );

    assert( tobii_enumerate_local_device_urls( api, url_receiver, &url_count ) == TOBII_ERROR_NO_ERROR );
    assert( url_count == 1 && first_url[ 0 ] != '\0' );
    assert( tobii_enumerate_local_device_urls( api, NULL, NULL ) == TOBII_ERROR_INVALID_PARAMETER );
    assert( tobii_enumerate_local_device_urls_ex( api, url_receiver, &url_count, 0 )
        == TOBII_ERROR_INVALID_PARAMETER );
    assert( tobii_enumerate_local_device_urls_ex( api, url_receiver, &url_count, TOBII_DEVICE_GENERATION_IS4 )
        == TOBII_ERROR_NO_ERROR );

    /* The system clock is CLOCK_MONOTONIC, as the daemon's host timestamps
     * are: it reads between two readings of that, and far behind the wall
     * clock. */
    int64_t const before = clock_us( CLOCK_MONOTONIC );
    int64_t t0 = 0, t1 = 0;
    assert( tobii_system_clock( api, &t0 ) == TOBII_ERROR_NO_ERROR );
    assert( tobii_system_clock( api, &t1 ) == TOBII_ERROR_NO_ERROR );
    int64_t const after = clock_us( CLOCK_MONOTONIC );
    assert( before <= t0 && t0 <= t1 && t1 <= after );
    assert( clock_us( CLOCK_REALTIME ) - t1 > far_from_the_wall_clock_us );

    /* An out-of-range field_of_use is refused without writing the handle —
     * this is what a caller built against the pre-4.0 header trips over. */
    tobii_device_t* device = NULL;
    assert( tobii_device_create( api, first_url, (tobii_field_of_use_t)0, &device )
        == TOBII_ERROR_INVALID_PARAMETER );
    assert( device == NULL );
    tobii_license_validation_result_t results[ 2 ] = { 99, 99 };
    assert( tobii_device_create_ex( api, first_url, (tobii_field_of_use_t)0, NULL, 2, results, &device )
        == TOBII_ERROR_INVALID_PARAMETER );
    assert( results[ 0 ] == 99 && device == NULL );

    /* Answered locally: the display area of a 597x336 mm monitor on the ET5
     * mounting is the one the Windows engine wrote (change-display.pcapng). */
    tobii_geometry_mounting_t const mounting = { 2, 184.0f, 20.0f, { 0.0f, -0.16f, 13.85f }, { 0.0f, 5.38f, 9.86f } };
    tobii_display_area_t area;
    assert( tobii_calculate_display_area_basic( api, 597.0f, 336.0f, 1.0022583f, &mounting, &area )
        == TOBII_ERROR_NO_ERROR );
    assert( fabsf( area.top_left_mm_xyz[ 0 ] + 297.49773f ) < 1e-3f );
    assert( fabsf( area.top_left_mm_xyz[ 1 ] - 326.00406f ) < 1e-3f );
    assert( fabsf( area.bottom_left_mm_xyz[ 2 ] + 3.10002f ) < 1e-3f );

    /* Unimplemented entry points answer, whatever they are given. */
    assert( tobii_license_key_store( NULL, NULL, 0 ) == TOBII_ERROR_NOT_SUPPORTED );
    assert( tobii_wearable_consumer_data_subscribe( NULL, NULL, NULL ) == TOBII_ERROR_NOT_SUPPORTED );
    assert( tobii_open_realm( NULL, 0, NULL, 0 ) == TOBII_ERROR_NOT_SUPPORTED );
    assert( tobii_calibration_collect_data_3d( NULL, 0.5f, 0.5f, 0.5f ) == TOBII_ERROR_NOT_SUPPORTED );

    /* Device-less calls are rejected before any device work. */
    assert( tobii_get_output_frequency( NULL, NULL ) == TOBII_ERROR_INVALID_PARAMETER );
    assert( tobii_calibration_start( NULL, TOBII_ENABLED_EYE_BOTH ) == TOBII_ERROR_INVALID_PARAMETER );
    assert( tobii_calibration_discard_data_2d( NULL, 0.5f, 0.5f ) == TOBII_ERROR_INVALID_PARAMETER );
    tobii_supported_t supported = TOBII_SUPPORTED;
    assert( tobii_internal_stream_supported( NULL, 0, &supported ) == TOBII_ERROR_INVALID_PARAMETER );
    tobii_lens_configuration_writable_t writable = TOBII_LENS_CONFIGURATION_WRITABLE;
    assert( tobii_lens_configuration_writable( NULL, &writable ) == TOBII_ERROR_INVALID_PARAMETER );
    assert( supported == TOBII_SUPPORTED && writable == TOBII_LENS_CONFIGURATION_WRITABLE );
    tobii_timesync_data_t timesync = { 1, 2, 3 };
    assert( tobii_timesync( NULL, &timesync ) == TOBII_ERROR_INVALID_PARAMETER );
    assert( timesync.system_start_us == 1 && timesync.tracker_us == 3 );
    assert( tobii_enumerate_stream_types( NULL, NULL, NULL ) == TOBII_ERROR_INVALID_PARAMETER );
    assert( tobii_pause_device( NULL ) == TOBII_ERROR_INVALID_PARAMETER );
    tobii_hardware_configuration_t hardware = { .entry_count = 7, .mode = 7 };
    assert( tobii_hardware_configuration_get( NULL, &hardware ) == TOBII_ERROR_INVALID_PARAMETER );
    assert( hardware.entry_count == 7 && hardware.mode == 7 );
    assert( tobii_resume_device( NULL ) == TOBII_ERROR_INVALID_PARAMETER );
    tobii_device_name_t name = "unchanged";
    assert( tobii_set_device_name( NULL, name ) == TOBII_ERROR_INVALID_PARAMETER );
    assert( tobii_get_device_name( NULL, &name ) == TOBII_ERROR_INVALID_PARAMETER );
    assert( strcmp( name, "unchanged" ) == 0 );
    assert( tobii_device_destroy( NULL ) == TOBII_ERROR_INVALID_PARAMETER );
    assert( tobii_api_destroy( NULL ) == TOBII_ERROR_INVALID_PARAMETER );

    assert( tobii_error_message( TOBII_ERROR_NO_ERROR ) != NULL );
    assert( tobii_error_message( (tobii_error_t)9999 ) != NULL );

    /* Enum numbering is the Stream Engine's. */
    assert( TOBII_ERROR_CONNECTION_FAILED == 5 && TOBII_ERROR_INVALID_PARAMETER == 8 );
    assert( TOBII_ERROR_CONFLICTING_API_INSTANCES == 14 && TOBII_ERROR_FIRMWARE_UPGRADE_IN_PROGRESS == 20 );
    assert( TOBII_CAPABILITY_COMPOUND_STREAM_WEARABLE_INCREASE_EYE_RELIEF == 18 );
    _Static_assert( TOBII_STREAM_GAZE_POINT == 0 && TOBII_STREAM_GAZE_ORIGIN == 1
            && TOBII_STREAM_EYE_POSITION_NORMALIZED == 2 && TOBII_STREAM_USER_PRESENCE == 3,
        "tobii_stream_t is numbered as the 4.1 DLL numbers it" );
    _Static_assert( TOBII_STREAM_HEAD_POSE == 4 && TOBII_STREAM_GAZE_DATA == 5
            && TOBII_STREAM_DIGITAL_SYNCPORT == 6 && TOBII_STREAM_DIAGNOSTICS_IMAGE == 7,
        "tobii_stream_t is numbered as the 4.1 DLL numbers it" );
    _Static_assert( TOBII_STREAM_USER_POSITION_GUIDE == 8 && TOBII_STREAM_WEARABLE_CONSUMER == 9
            && TOBII_STREAM_WEARABLE_ADVANCED == 10 && TOBII_STREAM_WEARABLE_FOVEATED_GAZE == 11,
        "tobii_stream_t is numbered as the 4.1 DLL numbers it" );
    assert( TOBII_STATE_CALIBRATION_ACTIVE == 7 );
    assert( TOBII_NOTIFICATION_TYPE_FACE_TYPE_CHANGED == 12 && TOBII_NOTIFICATION_VALUE_TYPE_STRING == 6 );
    assert( TOBII_LENS_CONFIGURATION_NOT_WRITABLE == 0 && TOBII_LENS_CONFIGURATION_WRITABLE == 1 );
    _Static_assert( TOBII_CALIBRATION_POINT_STATUS_FAILED_OR_INVALID == 0
            && TOBII_CALIBRATION_POINT_STATUS_VALID_BUT_NOT_USED_IN_CALIBRATION == 1
            && TOBII_CALIBRATION_POINT_STATUS_VALID_AND_USED_IN_CALIBRATION == 2,
        "numbered as tobii_calibration_parse writes them" );

    /* Struct layouts are shared memory between this program and the library;
     * the same numbers are pinned in the Rust tests. */
    assert( sizeof( tobii_device_info_t ) == 2048 );
    assert( offsetof( tobii_device_info_t, model ) == 256 );
    assert( offsetof( tobii_device_info_t, runtime_build_version ) == 0x700 );
    assert( sizeof( tobii_track_box_t ) == 96 );
    assert( sizeof( tobii_display_area_t ) == 36 );
    assert( sizeof( tobii_geometry_mounting_t ) == 36 );
    assert( sizeof( tobii_head_pose_t ) == 48 );
    assert( offsetof( tobii_head_pose_t, rotation_xyz ) == 36 );
    assert( sizeof( tobii_gaze_point_t ) == 24 );
    assert( sizeof( tobii_gaze_origin_t ) == 40 );
    assert( sizeof( tobii_eye_position_normalized_t ) == 40 );
    assert( sizeof( tobii_user_position_guide_t ) == 40 );
    assert( sizeof( tobii_gaze_data_eye_t ) == 76 );
    assert( sizeof( tobii_gaze_data_t ) == 168 );
    assert( offsetof( tobii_gaze_data_t, right ) == 92 );
    assert( sizeof( tobii_notification_t ) == 520 );
    assert( offsetof( tobii_notification_t, value ) == 8 );
    assert( sizeof( tobii_image_t ) == 32 );
    assert( offsetof( tobii_image_t, data ) == 24 );
    assert( sizeof( tobii_timesync_data_t ) == 24 );
    assert( offsetof( tobii_timesync_data_t, system_end_us ) == 8 );
    assert( offsetof( tobii_timesync_data_t, tracker_us ) == 16 );
    assert( sizeof( tobii_stream_type_t ) == 136 );
    assert( offsetof( tobii_stream_type_t, value ) == 4 );
    assert( offsetof( tobii_stream_type_t, name ) == 8 );
    assert( offsetof( tobii_stream_type_t, text ) == 72 );
    assert( sizeof( tobii_hardware_configuration_entry_t ) == 0x2e8 );
    assert( offsetof( tobii_hardware_configuration_entry_t, position_xyz ) == 0x10 );
    assert( offsetof( tobii_hardware_configuration_entry_t, width ) == 0xa0 );
    assert( offsetof( tobii_hardware_configuration_entry_t, coefficients ) == 0xb0 );
    assert( offsetof( tobii_hardware_configuration_entry_t, param_d ) == 0x2e0 );
    assert( sizeof( tobii_hardware_configuration_t ) == 2472 );
    assert( offsetof( tobii_hardware_configuration_t, entries ) == 8 );
    assert( offsetof( tobii_hardware_configuration_t, point_count ) == 0x5d8 );
    assert( offsetof( tobii_hardware_configuration_t, points_xyz ) == 0x5e0 );
    assert( offsetof( tobii_hardware_configuration_t, mode ) == 0x9a0 );
    assert( sizeof( tobii_calibration_point_data_t ) == 32 );
    assert( sizeof( tobii_license_key_t ) == 16 );
    assert( sizeof( tobii_device_name_t ) == 64 );
    assert( sizeof( tobii_state_string_t ) == 512 );

    assert( tobii_api_destroy( api ) == TOBII_ERROR_NO_ERROR );

    printf( "abi-smoke: ok (%zu symbols, device url \"%s\")\n", exported, first_url );
    return 0;
}
