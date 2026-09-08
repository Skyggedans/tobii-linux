//! Thin client: subscribe to gaze from `tobiid` (gaze mode) and, while the user
//! looks at the left or right edge of the screen, emit a Left/Right arrow key
//! once per second via the kernel `uinput` device. Works under both X11 and
//! Wayland because the key events are injected at the kernel input layer.
//!
//! Needs write access to `/dev/uinput` (typically root): run with `sudo`, or
//! grant your user access to the device.

use std::cmp::Ordering;
use std::str::FromStr;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use tobii::ipc::{
    STREAM_GAZE, ServerMsg, decode_server, encode_subscribe, read_frame, write_frame,
};

mod uinput {
    //! Minimal dependency-free `uinput` keyboard: a `File` on `/dev/uinput`
    //! driven with raw libc ioctls.
    use std::fs::{File, OpenOptions};
    use std::io::{self, Write};
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::AsRawFd;
    use std::time::Duration;

    // ioctl request numbers (see <linux/uinput.h>, computed for x86_64).
    const UI_SET_EVBIT: libc::c_ulong = 0x4004_5564; // _IOW('U', 100, int)
    const UI_SET_KEYBIT: libc::c_ulong = 0x4004_5565; // _IOW('U', 101, int)
    const UI_DEV_CREATE: libc::c_ulong = 0x5501; // _IO('U', 1)
    const UI_DEV_DESTROY: libc::c_ulong = 0x5502; // _IO('U', 2)

    const EV_SYN: u16 = 0x00;
    const EV_KEY: u16 = 0x01;
    const SYN_REPORT: u16 = 0;
    const BUS_USB: u16 = 0x03;

    /// evdev key code of the Left arrow.
    pub const KEY_LEFT: u16 = 105;
    /// evdev key code of the Right arrow.
    pub const KEY_RIGHT: u16 = 106;

    /// `struct input_id` from `<linux/input.h>`.
    #[repr(C)]
    struct InputId {
        bustype: u16,
        vendor: u16,
        product: u16,
        version: u16,
    }

    /// `struct uinput_user_dev` from `<linux/uinput.h>`.
    #[repr(C)]
    struct UinputUserDev {
        name: [u8; 80],
        id: InputId,
        ff_effects_max: u32,
        absmax: [i32; 64],
        absmin: [i32; 64],
        absfuzz: [i32; 64],
        absflat: [i32; 64],
    }

    /// `struct input_event` on 64-bit Linux: 16-byte `timeval`, type, code, value.
    #[repr(C)]
    struct InputEvent {
        tv_sec: i64,
        tv_usec: i64,
        type_: u16,
        code: u16,
        value: i32,
    }

    /// Marker for `#[repr(C)]` structs that can be handed to the kernel as raw
    /// bytes.
    ///
    /// # Safety
    ///
    /// Implementors must be `#[repr(C)]`, contain only integer fields (and
    /// arrays/structs of them) and have no padding, so every byte of a value
    /// is initialised.
    unsafe trait PlainBytes {}

    // SAFETY: `#[repr(C)]`; 80 + 4*2 + 4 + 4*64*4 bytes of integers, all
    // naturally aligned, no padding.
    unsafe impl PlainBytes for UinputUserDev {}
    // SAFETY: `#[repr(C)]`; 8 + 8 + 2 + 2 + 4 = 24 bytes of integers, no padding.
    unsafe impl PlainBytes for InputEvent {}

    /// View a kernel ABI struct as the bytes the driver expects on `write(2)`.
    fn as_bytes<T: PlainBytes>(v: &T) -> &[u8] {
        // SAFETY: `PlainBytes` guarantees `T` has no padding or non-integer
        // fields, so all `size_of::<T>()` bytes behind `v` are initialised; the
        // borrow keeps them alive and unaliased-mutably for the returned lifetime.
        unsafe { std::slice::from_raw_parts(std::ptr::from_ref(v).cast::<u8>(), size_of::<T>()) }
    }

    /// `ioctl(fd, req, arg)` for the `_IOW(int)`-style uinput requests.
    fn ioctl_int(file: &File, req: libc::c_ulong, arg: libc::c_int) -> io::Result<()> {
        // SAFETY: `file` keeps the descriptor open for the duration of the call
        // and the request takes a plain `int` by value — no pointer is passed.
        let rc = unsafe { libc::ioctl(file.as_raw_fd(), req, arg) };
        if rc < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// `ioctl(fd, req)` for the argument-less `_IO`-style uinput requests.
    fn ioctl_none(file: &File, req: libc::c_ulong) -> io::Result<()> {
        // SAFETY: `file` keeps the descriptor open for the duration of the call
        // and the request takes no argument.
        let rc = unsafe { libc::ioctl(file.as_raw_fd(), req) };
        if rc < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// A virtual keyboard registered with the kernel; unregistered on drop.
    pub struct Keyboard {
        file: File,
    }

    impl Keyboard {
        /// Create a virtual keyboard exposing the given keys.
        pub fn new(keys: &[u16]) -> io::Result<Self> {
            let file = OpenOptions::new()
                .write(true)
                .custom_flags(libc::O_NONBLOCK)
                .open("/dev/uinput")?;
            // From here on `Drop` issues UI_DEV_DESTROY on any early return.
            let kb = Keyboard { file };

            ioctl_int(&kb.file, UI_SET_EVBIT, libc::c_int::from(EV_KEY))?;
            ioctl_int(&kb.file, UI_SET_EVBIT, libc::c_int::from(EV_SYN))?;
            for &k in keys {
                ioctl_int(&kb.file, UI_SET_KEYBIT, libc::c_int::from(k))?;
            }

            let mut dev = UinputUserDev {
                name: [0; 80],
                id: InputId {
                    bustype: BUS_USB,
                    vendor: 0x1234,
                    product: 0x5678,
                    version: 1,
                },
                ff_effects_max: 0,
                absmax: [0; 64],
                absmin: [0; 64],
                absfuzz: [0; 64],
                absflat: [0; 64],
            };
            let name = b"tobii-gaze-keys";
            dev.name[..name.len()].copy_from_slice(name);
            (&kb.file).write_all(as_bytes(&dev))?;
            ioctl_none(&kb.file, UI_DEV_CREATE)?;
            // Give udev/compositor a moment to register the new device.
            std::thread::sleep(Duration::from_millis(200));
            Ok(kb)
        }

        fn emit(&self, type_: u16, code: u16, value: i32) -> io::Result<()> {
            let ev = InputEvent {
                tv_sec: 0,
                tv_usec: 0,
                type_,
                code,
                value,
            };
            (&self.file).write_all(as_bytes(&ev))
        }

        /// Press and release one key.
        pub fn tap(&self, key: u16) -> io::Result<()> {
            self.emit(EV_KEY, key, 1)?;
            self.emit(EV_SYN, SYN_REPORT, 0)?;
            self.emit(EV_KEY, key, 0)?;
            self.emit(EV_SYN, SYN_REPORT, 0)?;
            Ok(())
        }
    }

    impl Drop for Keyboard {
        fn drop(&mut self) {
            // The descriptor itself is closed when `file` drops right after this.
            if let Err(e) = ioctl_none(&self.file, UI_DEV_DESTROY) {
                tracing::warn!(error = %e, "UI_DEV_DESTROY failed");
            }
        }
    }
}

mod superkey {
    //! Track whether a Super (Meta) key is physically held, by reading evdev
    //! `/dev/input/event*` in background threads. Works under X11 and Wayland.
    use std::fs;
    use std::io::Read;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    const EV_KEY: u16 = 0x01;
    const KEY_LEFTMETA: u16 = 125;
    const KEY_RIGHTMETA: u16 = 126;

    /// Live state of the two Meta keys, fed by one reader thread per device.
    pub struct Watcher {
        left: Arc<AtomicBool>,
        right: Arc<AtomicBool>,
    }

    impl Watcher {
        /// Open every readable input device and watch it for Meta key events.
        pub fn spawn() -> std::io::Result<Self> {
            let left = Arc::new(AtomicBool::new(false));
            let right = Arc::new(AtomicBool::new(false));
            let mut opened = 0;
            for entry in fs::read_dir("/dev/input")? {
                let entry = entry?;
                if !entry.file_name().to_string_lossy().starts_with("event") {
                    continue;
                }
                // Some nodes aren't readable by us (or are busy) — skip them.
                let Ok(file) = fs::File::open(entry.path()) else {
                    continue;
                };
                opened += 1;
                let (l, r) = (Arc::clone(&left), Arc::clone(&right));
                thread::spawn(move || reader_loop(file, &l, &r));
            }
            if opened == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "no readable /dev/input/event* devices (join the `input` group?)",
                ));
            }
            Ok(Watcher { left, right })
        }

        /// True while either Super/Meta key is currently down.
        pub fn is_down(&self) -> bool {
            // Relaxed: the flags are pure signals; no other data is published
            // through them.
            self.left.load(Ordering::Relaxed) || self.right.load(Ordering::Relaxed)
        }
    }

    fn reader_loop(mut file: fs::File, left: &AtomicBool, right: &AtomicBool) {
        // input_event on 64-bit: 16B timeval + u16 type + u16 code + i32 value.
        let mut buf = [0u8; 24];
        while file.read_exact(&mut buf).is_ok() {
            if u16::from_ne_bytes([buf[16], buf[17]]) != EV_KEY {
                continue;
            }
            let code = u16::from_ne_bytes([buf[18], buf[19]]);
            let value = i32::from_ne_bytes([buf[20], buf[21], buf[22], buf[23]]);
            let down = value != 0; // 1=press, 2=autorepeat, 0=release
            match code {
                KEY_LEFTMETA => left.store(down, Ordering::Relaxed),
                KEY_RIGHTMETA => right.store(down, Ordering::Relaxed),
                _ => {}
            }
        }
    }
}

fn main() {
    tobii::logging::init();
    if let Err(e) = run() {
        eprintln!("tobii-gaze-keys: {e:#}");
        std::process::exit(1);
    }
}

/// Take and parse the value following `flag`.
fn next_value<T>(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<T>
where
    T: FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let v = args
        .next()
        .with_context(|| format!("{flag} needs a value"))?;
    v.parse()
        .with_context(|| format!("{flag}: invalid value {v:?}"))
}

fn run() -> Result<()> {
    // Manual thresholds (absolute gaze-X). Left unset => adaptive auto-calibration,
    // which tracks the actually-observed gaze-X range and triggers near its edges.
    // This handles trackers whose gaze-X isn't centered/symmetric (a fixed 0.15
    // left edge may simply be unreachable while 0.85 right is).
    let mut left_edge: Option<f32> = None;
    let mut right_edge: Option<f32> = None;
    let mut margin = 0.25f32; // auto: zone width as a fraction of the observed span
    let mut interval = Duration::from_millis(500);
    let mut require_super = true; // only tap while a Super/Meta key is held
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--left" => left_edge = Some(next_value(&mut args, "--left")?),
            "--right" => right_edge = Some(next_value(&mut args, "--right")?),
            "--margin" => margin = next_value(&mut args, "--margin")?,
            "--no-super" => require_super = false,
            "--interval-ms" => {
                interval = Duration::from_millis(next_value(&mut args, "--interval-ms")?);
            }
            "-h" | "--help" => {
                println!(
                    "usage: tobii-gaze-keys [--margin 0.25] [--interval-ms 1000] [--left X --right X] [--no-super]\n\
                     \n\
                     Watches gaze X. While gaze is at the left/right edge of the screen\n\
                     AND a Super/Meta key is held, taps the Left/Right arrow key once per\n\
                     interval. Pass --no-super to drop the Super requirement. Needs\n\
                     /dev/uinput + /dev/input read access (sudo, or the `input` group).\n\
                     \n\
                     Default = AUTO calibration: it learns the gaze-X range you actually\n\
                     reach (look fully left and fully right once) and fires when gaze is\n\
                     within --margin of either observed extreme. Robust to a gaze stream\n\
                     that isn't centered/symmetric.\n\
                     \n\
                     --left/--right switch to FIXED absolute thresholds (0=left .. 1=right)."
                );
                return Ok(());
            }
            other => bail!("unknown arg: {other}"),
        }
    }
    // Fixed thresholds as soon as either side is given (the other side gets a
    // sane default); otherwise auto calibration.
    let edges = match (left_edge, right_edge) {
        (None, None) => None,
        (l, r) => {
            let (l, r) = (l.unwrap_or(0.15), r.unwrap_or(0.85));
            // `partial_cmp` so that a NaN on either side is rejected as well.
            ensure!(
                l.partial_cmp(&r) == Some(Ordering::Less),
                "--left must be less than --right"
            );
            Some((l, r))
        }
    };
    ensure!(
        margin > 0.0 && margin < 0.5,
        "--margin must be between 0 and 0.5"
    );

    let kb = uinput::Keyboard::new(&[uinput::KEY_LEFT, uinput::KEY_RIGHT])
        .context("opening /dev/uinput (try running with sudo)")?;

    // Watch the physical Super/Meta key (unless gating is disabled).
    let supers = if require_super {
        Some(
            superkey::Watcher::spawn()
                .context("watching modifier keys (need read access to /dev/input/event*)")?,
        )
    } else {
        None
    };

    let mut stream = tobii::ipc::connect_or_spawn().context("connecting to tobiid")?;
    write_frame(&mut stream, &encode_subscribe(STREAM_GAZE))
        .context("subscribing to the gaze stream")?;

    let gate = if require_super {
        " while Super held"
    } else {
        ""
    };
    if let Some((l, r)) = edges {
        println!(
            "tobii-gaze-keys: fixed edges <{l:.2} / >{r:.2}, 1 tap / {}ms{gate}",
            interval.as_millis()
        );
    } else {
        println!(
            "tobii-gaze-keys: AUTO calibration (margin {margin:.2}), 1 tap / {}ms{gate}\n\
             look fully LEFT and fully RIGHT once to learn your gaze range...",
            interval.as_millis()
        );
    }

    // Rate-limit each direction independently; fire immediately on entering a zone.
    let mut last_left: Option<Instant> = None;
    let mut last_right: Option<Instant> = None;

    // Adaptive envelope of observed gaze-X (auto mode). We need at least this much
    // span before triggering, so a stationary gaze can't fire from noise alone.
    const MIN_SPAN: f32 = 0.20;
    let mut xmin = f32::INFINITY;
    let mut xmax = f32::NEG_INFINITY;

    loop {
        let Some(body) = read_frame(&mut stream).context("reading from tobiid")? else {
            bail!("daemon closed the connection");
        };
        match decode_server(&body) {
            Some(ServerMsg::Subscribed { ok: false }) => {
                bail!("daemon is busy with the other mode (head)")
            }
            Some(ServerMsg::Subscribed { ok: true }) => {}
            Some(ServerMsg::Gaze { valid, xy, .. }) => {
                let now = Instant::now();
                if !valid {
                    // Lost gaze: reset so re-entry taps right away.
                    last_left = None;
                    last_right = None;
                    continue;
                }
                let x = xy[0];

                // Resolve the active left/right thresholds for this sample.
                let (lo, hi, calibrating) = if let Some((l, r)) = edges {
                    (l, r, false)
                } else {
                    // Auto: grow the envelope, then relax it gently toward the
                    // center so a one-off outlier doesn't pin an extreme forever.
                    xmin = xmin.min(x);
                    xmax = xmax.max(x);
                    let c = 0.5 * (xmin + xmax);
                    xmin += (c - xmin) * 0.0008;
                    xmax += (c - xmax) * 0.0008;
                    let span = xmax - xmin;
                    if span < MIN_SPAN {
                        (f32::NEG_INFINITY, f32::INFINITY, true) // not ready: no zone
                    } else {
                        (xmin + margin * span, xmax - margin * span, false)
                    }
                };

                // Gate on the physical Super/Meta key, if required.
                let super_held = supers.as_ref().is_none_or(superkey::Watcher::is_down);

                if calibrating {
                    last_left = None;
                    last_right = None;
                    eprint!("\r   calibrating (x={x:.2}, range {xmin:.2}..{xmax:.2})        ");
                } else if !super_held {
                    // Idle until Super is pressed; reset so re-entry taps right away.
                    last_left = None;
                    last_right = None;
                    eprint!("\r   hold Super  (x={x:.2})        ");
                } else if x <= lo {
                    last_right = None;
                    if last_left.is_none_or(|t| now.duration_since(t) >= interval) {
                        kb.tap(uinput::KEY_LEFT).context("tapping Left")?;
                        last_left = Some(now);
                        eprint!("\r<- LEFT  (x={x:.2})        ");
                    }
                } else if x >= hi {
                    last_left = None;
                    if last_right.is_none_or(|t| now.duration_since(t) >= interval) {
                        kb.tap(uinput::KEY_RIGHT).context("tapping Right")?;
                        last_right = Some(now);
                        eprint!("\r-> RIGHT (x={x:.2})        ");
                    }
                } else {
                    // Center zone: idle, ready to fire on next edge entry.
                    last_left = None;
                    last_right = None;
                    eprint!("\r   center (x={x:.2})        ");
                }
            }
            // Head/presence frames aren't subscribed here and unknown tags decode
            // to `None`. `ServerMsg` belongs to the library crate (this binary is
            // a separate crate), so a wildcard keeps this client building if the
            // enum grows or becomes `#[non_exhaustive]`.
            _ => {}
        }
    }
}
