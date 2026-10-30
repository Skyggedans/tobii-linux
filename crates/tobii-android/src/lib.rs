//! JNI bridge: the Tobii Eye Tracker 5's gaze stream in an Android app,
//! using the file descriptor `android.hardware.usb.UsbManager`'s own
//! permission dialog hands over, no root required.
//!
//! A minimal proof of concept, not the daemon/client split the Linux and
//! macOS builds use: one engine runs inside the app's own process, and a
//! background thread drains its samples into a single-slot latest-gaze
//! store that [`latestGaze`] reads back. The Kotlin side is expected to
//! poll it on a timer.

mod calibration;

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use jni::JNIEnv;
use jni::objects::{JClass, JString};
use jni::sys::{jboolean, jdoubleArray, jint, jstring};
use tobii_usb::engine::{Engine, Sample};

/// The last calibration call's error, if its last call failed: there is no
/// logcat route for `tracing`/`eprintln` wired up in this minimal bridge, so
/// [`lastCalibrationError`] is how Kotlin finds out why.
fn last_error_slot() -> &'static Mutex<String> {
    static LAST_ERROR: OnceLock<Mutex<String>> = OnceLock::new();
    LAST_ERROR.get_or_init(|| Mutex::new(String::new()))
}

fn note_error(context: &str, e: &str) {
    *last_error_slot().lock().unwrap() = format!("{context}: {e}");
}

#[derive(Clone, Copy, Default)]
struct LatestGaze {
    x: f64,
    y: f64,
    device_ts_us: i64,
    valid: bool,
}

fn engine_slot() -> &'static Mutex<Option<Engine>> {
    static ENGINE: OnceLock<Mutex<Option<Engine>>> = OnceLock::new();
    ENGINE.get_or_init(|| Mutex::new(None))
}

/// The running engine's command handle, for a JNI call that needs the
/// device: `None` with nothing logged when there is no engine, so a stray
/// calibration call before `startEngine` (or after `stopEngine`) just does
/// nothing.
fn commands() -> Option<tobii_usb::engine::Commands> {
    engine_slot().lock().unwrap().as_ref().map(Engine::commands)
}

fn latest_slot() -> &'static Mutex<LatestGaze> {
    static LATEST: OnceLock<Mutex<LatestGaze>> = OnceLock::new();
    LATEST.get_or_init(|| Mutex::new(LatestGaze::default()))
}

fn drain_loop() {
    loop {
        std::thread::sleep(Duration::from_millis(16));
        let mut guard = engine_slot().lock().unwrap();
        let Some(engine) = guard.as_mut() else { continue };
        for sample in engine.drain() {
            if let Sample::Gaze(g) = sample {
                *latest_slot().lock().unwrap() = LatestGaze {
                    x: g.frame.gaze.value[0],
                    y: g.frame.gaze.value[1],
                    device_ts_us: i64::try_from(g.frame.device_ts_us).unwrap_or(0),
                    valid: g.frame.gaze.valid,
                };
            }
        }
    }
}

/// Starts the engine from a file descriptor already open and permitted on
/// the Java side (`UsbManager`/`UsbDeviceConnection`), skipping libusb's own
/// enumeration/open and the USB baseline reset, neither of which makes
/// sense over a descriptor Android already owns (see
/// [`tobii_usb::device::open_tobii_with_fd`]). Safe to call more than once;
/// only the first call's drain thread keeps running.
///
/// # Safety
///
/// `fd` must be a valid, open file descriptor for exactly the Tobii Eye
/// Tracker 5 (vendor 0x2104, product 0x0313), kept open by the caller for as
/// long as the engine runs.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_tobiilinux_poc_NativeBridge_startEngine(
    _env: JNIEnv,
    _class: JClass,
    fd: jint,
) {
    // SAFETY: the caller's own contract above is exactly what
    // `TOBII_ANDROID_FD` requires of `open_tobii_with_fd`.
    unsafe {
        std::env::set_var("TOBII_ANDROID_FD", fd.to_string());
        std::env::set_var("TOBII_NO_RESET", "1");
    }
    *engine_slot().lock().unwrap() = Some(Engine::start());

    static DRAINER_STARTED: OnceLock<()> = OnceLock::new();
    DRAINER_STARTED.get_or_init(|| {
        std::thread::spawn(drain_loop);
    });
}

/// Stops and drops the engine, releasing the fd-wrapped device handle.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_tobiilinux_poc_NativeBridge_stopEngine(
    _env: JNIEnv,
    _class: JClass,
) {
    engine_slot().lock().unwrap().take();
}

/// The latest gaze point, normalised display coordinates:
/// `[x, y, validity (0.0/1.0), device timestamp (µs)]`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_tobiilinux_poc_NativeBridge_latestGaze(
    env: JNIEnv,
    _class: JClass,
) -> jdoubleArray {
    let g = *latest_slot().lock().unwrap();
    let values = [g.x, g.y, f64::from(i32::from(g.valid)), g.device_ts_us as f64];
    let out = env.new_double_array(4).expect("new_double_array");
    env.set_double_array_region(&out, 0, &values)
        .expect("set_double_array_region");
    out.into_raw()
}

/// Starts a calibration session (see [`calibration::start`]). `storage_dir`
/// is the app's own files directory (`context.filesDir.absolutePath`),
/// where the result is saved and loaded back from on the next session.
/// `false` on any failure (logged to stderr, which `adb logcat` carries);
/// there is no session to collect points into then.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_tobiilinux_poc_NativeBridge_calibrationStart(
    mut env: JNIEnv,
    _class: JClass,
    storage_dir: JString,
) -> jboolean {
    let Ok(storage_dir) = env.get_string(&storage_dir) else {
        return jboolean::from(false);
    };
    let storage_dir: String = storage_dir.into();
    let Some(commands) = commands() else {
        return jboolean::from(false);
    };
    match calibration::start(&commands, &storage_dir) {
        Ok(()) => jboolean::from(true),
        Err(e) => {
            note_error("calibration start", &e);
            jboolean::from(false)
        }
    }
}

/// Collects one calibration point at `(x, y)`, normalised display
/// coordinates (0..=1 each way). `false` on any failure.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_tobiilinux_poc_NativeBridge_calibrationCollect(
    _env: JNIEnv,
    _class: JClass,
    x: jni::sys::jdouble,
    y: jni::sys::jdouble,
) -> jboolean {
    let Some(commands) = commands() else {
        return jboolean::from(false);
    };
    #[allow(clippy::cast_possible_truncation)]
    match calibration::collect(&commands, x as f32, y as f32) {
        Ok(()) => jboolean::from(true),
        Err(e) => {
            note_error("calibration collect", &e);
            jboolean::from(false)
        }
    }
}

/// Computes the calibration from the points collected so far. The new
/// calibration's id, or -1 on any failure (nothing collected, the device
/// refused, or what it computed does not validate).
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_tobiilinux_poc_NativeBridge_calibrationCompute(
    _env: JNIEnv,
    _class: JClass,
) -> jint {
    let Some(commands) = commands() else { return -1 };
    match calibration::compute(&commands) {
        #[allow(clippy::cast_possible_wrap)]
        Ok(id) => id as jint,
        Err(e) => {
            note_error("calibration compute", &e);
            -1
        }
    }
}

/// Ends the session: `keep` writes back and saves the last computed
/// calibration (if any validated); otherwise the one the session started
/// from goes back. Returns whether a calibration was saved.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_tobiilinux_poc_NativeBridge_calibrationStop(
    _env: JNIEnv,
    _class: JClass,
    keep: jboolean,
) -> jboolean {
    let Some(commands) = commands() else {
        return jboolean::from(false);
    };
    match calibration::stop(&commands, keep != 0) {
        Ok(saved) => jboolean::from(saved),
        Err(e) => {
            note_error("calibration stop", &e);
            jboolean::from(false)
        }
    }
}

/// Why the last calibration call that failed did: empty when none has.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_tobiilinux_poc_NativeBridge_lastCalibrationError(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    let message = last_error_slot().lock().unwrap().clone();
    env.new_string(message)
        .expect("new_string")
        .into_raw()
}
