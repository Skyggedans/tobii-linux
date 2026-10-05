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
 * Threads: as the Stream Engine promises, the functions may be called from
 * several threads at once, on one device too. A device's state is locked by
 * concern, as the DLL's is:
 * - its requests (device info, states, calibration, pause and the other
 *   calls answered by tobiid), subscribes, unsubscribes and
 *   tobii_device_reconnect run one at a time, in the order they are called,
 *   each for its whole round trip to tobiid, so a slow one (a calibration
 *   start may take ~3 min at worst, a pause a minute) delays the others on
 *   that device, and one thread's calls made back to back hold another
 *   thread's up for one of them at most; of them only a reconnect holds
 *   back its callbacks, tobii_device_process_callbacks and
 *   tobii_wait_for_callbacks, for its own round trip (~500 ms at most) and
 *   the close of its old connection;
 *   tobii_recenter, a write with no reply, waits for those under way or
 *   called before it;
 * - its callbacks run one at a time, on whichever thread calls
 *   tobii_device_process_callbacks, and such a call made while another
 *   thread processes the device returns at once;
 * - a subscribe, an unsubscribe, tobii_device_clear_callback_buffers and
 *   tobii_device_reconnect wait for a callback another thread is running
 *   (the last two for that thread's whole process call), and once an
 *   unsubscribe returns, its callback is not running and never runs again
 *   (tobii_streams.h says more);
 * - tobii_device_destroy and tobii_api_destroy take no lock, as in the
 *   Stream Engine: no other thread may be inside a call on the handle, a
 *   tobii_wait_for_callbacks waiting on the device included, and none may
 *   use it afterwards. Join the thread that processes a device first.
 * TOBII_ERROR_CALLBACK_IN_PROGRESS guards only the thread a callback, the
 * logger or tobii_calibration_retrieve's receiver runs on; other threads'
 * calls go on. A callback, or the logger, must not block on another thread's
 * call into any device (nor on a thread that waits for one), which can
 * deadlock, as in the Stream Engine: only tobii_device_process_callbacks and
 * tobii_wait_for_callbacks are sure to return while a callback runs; any
 * other call on its device may wait for it, itself or queued behind one that
 * does, and a call on another device may wait for that device's own
 * callback, which may be waiting in turn. The retrieve receiver holds no
 * lock, so it may wait for other threads' calls (tobii_config.h).
 * Where libtobii.so differs from the DLL, besides what the functions below
 * say: a device's requests, subscription changes and reconnects run in the
 * order they are called, where the DLL's critical section promises no order
 * among its waiters (Windows semantics, not read from the DLL), so there one
 * thread's calls made back to back may keep another thread's out for long;
 * tobii_wait_for_callbacks waits on a device another thread is processing as
 * on any other, where the DLL skips such a device, returning at once when it
 * was the only one; a subscribe lets the device's other callbacks run during
 * its round trip, where the DLL holds them back; tobii_calibration_retrieve
 * calls its receiver with no lock held, where the DLL holds the device's API
 * mutex; and the logger is never called under a lock of the call that logs.
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

/* tobii_api_create checks custom_alloc as the Stream Engine does: without
 * malloc_func or free_func it is TOBII_ERROR_INVALID_PARAMETER, and *api is
 * left as it was. The allocator is then never called: libtobii.so allocates
 * with Rust's allocator, so TOBII_ERROR_ALLOCATION_FAILED never comes from
 * it. Pass NULL. */
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

/* The only way to see libtobii.so's diagnostics: it prints nothing, and
 * nothing outside it can subscribe to its internal tracing. A custom_log
 * without log_func is TOBII_ERROR_INVALID_PARAMETER, as in the Stream Engine;
 * with custom_log NULL nothing is logged. log_func gets libtobii's own
 * diagnostics, not a line per failing call (the returned error says that):
 * TOBII_LOG_LEVEL_ERROR when field_of_use is refused, a device cannot connect
 * to tobiid or reconnect, tobii_device_process_callbacks reports a lost
 * connection (once per loss), tobiid sends a reply that does not decode, or
 * tobii_calibration_stop fails after tobiid saved the calibration (saved but
 * perhaps not applied: the tracker loads it at its next init);
 * TOBII_LOG_LEVEL_WARN, once per device, when tobiid has sent no head pose
 * in the 20 s since it was subscribed (a tobiid older than libtobii.so
 * sends none, tobii_streams.h); TOBII_LOG_LEVEL_INFO when a device connects
 * or reconnects.
 *
 * It is called synchronously, on the thread inside the tobii_* call that
 * logs, never on a thread of libtobii's own, with none of the locks that call
 * took held (a call made from inside a callback logs under that callback's),
 * and not serialised: threads may log at once, through one device too, so
 * log_func must be safe to call that way, as in the Stream Engine, and lines
 * from different threads may interleave (a loss one thread's
 * tobii_device_process_callbacks reports can come after a reconnect another
 * thread made since). A call from inside it that a stream callback could not
 * make either returns TOBII_ERROR_CALLBACK_IN_PROGRESS, on that thread only:
 * another thread it hands the line to may use any device meanwhile, but the
 * logger, like a callback, must not block on that thread's call into one
 * (see Threads at the top). text is valid only during the call. log_func
 * and log_context must stay valid until the API and every device created
 * from it are destroyed: a device keeps logging after tobii_api_destroy. */
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

/* The 4.1 DLL's texts. Never NULL, never to be freed, valid for the lifetime
 * of the process. An out-of-range code's text, a negative code's included, is
 * formatted into one buffer the process shares, which the next out-of-range
 * call, on any thread, rewrites, as in the DLL. */
TOBII_API char const* TOBII_CALL tobii_error_message( tobii_error_t error );

/* 4.1.0.3: the version this library imitates. */
TOBII_API tobii_error_t TOBII_CALL tobii_get_api_version( tobii_version_t* version );

TOBII_API tobii_error_t TOBII_CALL tobii_api_create( tobii_api_t** api,
    tobii_custom_alloc_t const* custom_alloc, tobii_custom_log_t const* custom_log );
/* Takes no lock, as in the Stream Engine: no other thread may be inside a
 * call on the API handle, and none may use it afterwards. The devices
 * created from it keep working, and keep logging to its logger. */
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
 * daemon cannot be reached (it is spawned on demand). Threads that create
 * devices at once while no daemon runs spawn one between them: the others
 * wait for that spawn, then connect to its daemon or fail as it did. */
TOBII_API tobii_error_t TOBII_CALL tobii_device_create( tobii_api_t* api, char const* url,
    tobii_field_of_use_t field_of_use, tobii_device_t** device );
/* Closes the daemon connection. Takes no lock, as in the Stream Engine: no
 * other thread may be inside a call on the device, a tobii_wait_for_callbacks
 * waiting on it included, and none may use it afterwards. */
TOBII_API tobii_error_t TOBII_CALL tobii_device_destroy( tobii_device_t* device );

/* Blocks until a device has a sample queued or a lost daemon connection not
 * yet reported by tobii_device_process_callbacks, or ~100 ms per idle device,
 * so a wait-and-process loop wakes once for a loss. A reported loss waits
 * like an idle device, unless the samples of a tobii_device_reconnect made on
 * another thread wake it. Never TOBII_ERROR_CONNECTION_FAILED. It holds no
 * lock while it sleeps, and waits on a device another thread is processing
 * as on any other, until that thread leaves it something to process; the DLL
 * skips such a device (read from its code, not observed), and with no other
 * returns TOBII_ERROR_NO_ERROR at once, which spins a wait-and-process
 * loop. */
TOBII_API tobii_error_t TOBII_CALL tobii_wait_for_callbacks( int device_count,
    tobii_device_t* const* devices );
/* Once the daemon connection is lost: delivers what had arrived, then returns
 * TOBII_ERROR_CONNECTION_FAILED on every call until tobii_device_reconnect
 * connects again; libtobii never reconnects by itself. A tracker unplug is
 * not reported here: the daemon keeps the connection, and the samples resume
 * on it after a replug. One thread dispatches a device at a time: a call
 * that finds another thread at it (processing, a wait or a clear at its
 * queue for a moment, or a reconnect waiting for tobiid to take the
 * subscriptions back, then closing the old connection) returns at once and
 * delivers nothing, leaving what is queued for the next call:
 * TOBII_ERROR_NO_ERROR, or TOBII_ERROR_CONNECTION_FAILED once the loss has
 * been reported. The DLL's returns TOBII_ERROR_NO_ERROR there even after a
 * loss, once it has delivered the device's queued notifications itself. */
TOBII_API tobii_error_t TOBII_CALL tobii_device_process_callbacks( tobii_device_t* device );
/* Drops what is queued; a lost connection is still reported by the next
 * process call. It waits for a tobii_device_process_callbacks, or a
 * tobii_device_reconnect's round trip and close of its old connection,
 * under way on another thread, never for a request; the DLL's waits for
 * requests and reconnects, under its API mutex, and for a callback another
 * thread is running, and while another thread processes it clears only the
 * queued notifications. */
TOBII_API tobii_error_t TOBII_CALL tobii_device_clear_callback_buffers( tobii_device_t* device );
/* Connects to a running daemon (never spawns one) and restores the
 * subscriptions, not a calibration session or pause. Any failure is
 * TOBII_ERROR_CONNECTION_FAILED within ~500 ms, counted from when the
 * requests, subscription changes and other reconnects other threads have in
 * flight or called before it, and then any tobii_device_process_callbacks
 * under way on another thread, have finished (it waits for them before it
 * asks for the subscriptions back), and leaves the device as it was. From
 * then until it has swapped connections or failed, a
 * tobii_device_process_callbacks on another thread returns at once and
 * delivers nothing. tobiid sends the new connection what it sends the old one
 * from its ack on, so of what the old one brought and was not delivered, the
 * samples are dropped: none is delivered twice, but a reconnect of a live
 * connection may lose those tobiid sent the old one alone. The old one's
 * notifications are kept, and delivered ahead of the new one's, as tobiid
 * sends each once and does not repeat it to a new connection; one sent to
 * both may be delivered again, after later ones, so a state may be seen to
 * step back before it settles: the last delivered is current. While the
 * daemon has no tracker (a request's TOBII_ERROR_CONNECTION_FAILED with the
 * connection intact) it succeeds without bringing one back. */
TOBII_API tobii_error_t TOBII_CALL tobii_device_reconnect( tobii_device_t* device );
/* A no-op: the daemon maps the tracker's clock to the host clock for every
 * sample and keeps the mapping current itself (tobii_streams.h says how), so
 * there is nothing here to refresh (tobii_timesync takes a fresh clock pair
 * on every call). */
TOBII_API tobii_error_t TOBII_CALL tobii_update_timesync( tobii_device_t* device );
/* CLOCK_MONOTONIC microseconds, whose epoch is undefined (the DLL reads
 * QueryPerformanceCounter): the clock of every callback timestamp (bar gaze
 * data's and raw gaze's timestamp_tracker_us) and of tobii_timesync's system
 * times. */
TOBII_API tobii_error_t TOBII_CALL tobii_system_clock( tobii_api_t* api, int64_t* timestamp_us );

/* Serial, model, generation, firmware and integration type, as the tracker
 * reports them; runtime_build_version names libtobii.so. integration_id,
 * hw_calibration_version, hw_calibration_date and lot_id are empty, as the
 * DLL leaves them for a tracker it drives over USB. From a tobiid that
 * predates it, the integration type is empty too. Fetched once per
 * connection; a read of the kept copy, too, waits for the requests other
 * threads have in flight, as in the DLL, or called before it. */
TOBII_API tobii_error_t TOBII_CALL tobii_get_device_info( tobii_device_t* device,
    tobii_device_info_t* device_info );
TOBII_API tobii_error_t TOBII_CALL tobii_get_track_box( tobii_device_t* device,
    tobii_track_box_t* track_box );

/* Bool states 0, 1 and 3 are always false, and so are FAULT and WARNING,
 * which the DLL refuses as bool states (TOBII_ERROR_INVALID_PARAMETER);
 * DEVICE_PAUSED and CALIBRATION_ACTIVE come from the daemon. Unlike the DLL,
 * DEVICE_PAUSED changes as soon as the tracker accepts a pause or resume,
 * and a tracker re-init ends a pause. The uint32 state is CALIBRATION_ID
 * (DEVICE_PAUSED there is TOBII_ERROR_INVALID_PARAMETER, as in the DLL).
 * The string states FAULT and WARNING are the tracker's fault and warning
 * lists ("ok" when there are none) as its last init reported them (command
 * 1490), or as it announced them since (FAULTS_CHANGED and WARNINGS_CHANGED
 * notifications; a daemon older than this answers the init's list and sends
 * neither), cut to 511 bytes (the DLL's own copy of the init's seems to hold
 * 119 bytes); TOBII_ERROR_NOT_SUPPORTED, with the value untouched, when that
 * init reported none, whatever was announced since (as the DLL), or from a
 * daemon too old to know these states. Unlike the DLL, which answers from
 * its cache, a string read can also be TOBII_ERROR_TIMED_OUT if the daemon
 * has not seen a tracker yet (the call waits for its first init) or
 * TOBII_ERROR_CONNECTION_FAILED when the daemon is gone. */
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
