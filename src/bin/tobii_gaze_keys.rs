//! Thin client: subscribe to gaze from `tobiid` (gaze mode) and, while the user
//! looks at the left or right edge of the screen, emit a Left/Right arrow key
//! once per second via the kernel `uinput` device. Works under both X11 and
//! Wayland because the key events are injected at the kernel input layer.
//!
//! Needs write access to `/dev/uinput` (typically root): run with `sudo`, or
//! grant your user access to the device.

use std::time::{Duration, Instant};

use tobii::ipc::{decode_server, encode_subscribe, read_frame, write_frame, ServerMsg, STREAM_GAZE};

mod uinput {
    //! Minimal dependency-free `uinput` keyboard via raw libc ioctl/write.
    use std::io;
    use std::os::unix::io::RawFd;

    // ioctl request numbers (see <linux/uinput.h>, computed for x86_64).
    const UI_SET_EVBIT: libc::c_ulong = 0x4004_5564; // _IOW('U', 100, int)
    const UI_SET_KEYBIT: libc::c_ulong = 0x4004_5565; // _IOW('U', 101, int)
    const UI_DEV_CREATE: libc::c_ulong = 0x5501; // _IO('U', 1)
    const UI_DEV_DESTROY: libc::c_ulong = 0x5502; // _IO('U', 2)

    const EV_SYN: u16 = 0x00;
    const EV_KEY: u16 = 0x01;
    const SYN_REPORT: u16 = 0;
    const BUS_USB: u16 = 0x03;

    pub const KEY_LEFT: u16 = 105;
    pub const KEY_RIGHT: u16 = 106;

    #[repr(C)]
    struct InputId {
        bustype: u16,
        vendor: u16,
        product: u16,
        version: u16,
    }

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

    #[repr(C)]
    struct InputEvent {
        tv_sec: i64,
        tv_usec: i64,
        type_: u16,
        code: u16,
        value: i32,
    }

    pub struct Keyboard {
        fd: RawFd,
    }

    impl Keyboard {
        /// Create a virtual keyboard exposing the given keys.
        pub fn new(keys: &[u16]) -> io::Result<Self> {
            let path = b"/dev/uinput\0";
            let fd = unsafe {
                libc::open(
                    path.as_ptr() as *const libc::c_char,
                    libc::O_WRONLY | libc::O_NONBLOCK,
                )
            };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            let kb = Keyboard { fd };

            unsafe {
                if libc::ioctl(fd, UI_SET_EVBIT, EV_KEY as libc::c_int) < 0 {
                    return Err(io::Error::last_os_error());
                }
                if libc::ioctl(fd, UI_SET_EVBIT, EV_SYN as libc::c_int) < 0 {
                    return Err(io::Error::last_os_error());
                }
                for &k in keys {
                    if libc::ioctl(fd, UI_SET_KEYBIT, k as libc::c_int) < 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
            }

            let mut dev: UinputUserDev = unsafe { std::mem::zeroed() };
            let name = b"tobii-gaze-keys";
            dev.name[..name.len()].copy_from_slice(name);
            dev.id = InputId {
                bustype: BUS_USB,
                vendor: 0x1234,
                product: 0x5678,
                version: 1,
            };
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &dev as *const _ as *const u8,
                    std::mem::size_of::<UinputUserDev>(),
                )
            };
            if unsafe { libc::write(fd, bytes.as_ptr() as *const libc::c_void, bytes.len()) }
                != bytes.len() as isize
            {
                return Err(io::Error::last_os_error());
            }
            if unsafe { libc::ioctl(fd, UI_DEV_CREATE) } < 0 {
                return Err(io::Error::last_os_error());
            }
            // Give udev/compositor a moment to register the new device.
            std::thread::sleep(std::time::Duration::from_millis(200));
            Ok(kb)
        }

        fn emit(&self, type_: u16, code: u16, value: i32) -> io::Result<()> {
            let ev = InputEvent { tv_sec: 0, tv_usec: 0, type_, code, value };
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &ev as *const _ as *const u8,
                    std::mem::size_of::<InputEvent>(),
                )
            };
            let n = unsafe {
                libc::write(self.fd, bytes.as_ptr() as *const libc::c_void, bytes.len())
            };
            if n != bytes.len() as isize {
                return Err(io::Error::last_os_error());
            }
            Ok(())
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
            unsafe {
                libc::ioctl(self.fd, UI_DEV_DESTROY);
                libc::close(self.fd);
            }
        }
    }
}

mod superkey {
    //! Track whether a Super (Meta) key is physically held, by reading evdev
    //! `/dev/input/event*` in background threads. Works under X11 and Wayland.
    use std::fs;
    use std::io::Read;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;

    const EV_KEY: u16 = 0x01;
    const KEY_LEFTMETA: u16 = 125;
    const KEY_RIGHTMETA: u16 = 126;

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
                let Ok(file) = fs::File::open(entry.path()) else { continue };
                opened += 1;
                let (l, r) = (Arc::clone(&left), Arc::clone(&right));
                thread::spawn(move || reader_loop(file, l, r));
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
        pub fn down(&self) -> bool {
            self.left.load(Ordering::Relaxed) || self.right.load(Ordering::Relaxed)
        }
    }

    fn reader_loop(mut file: fs::File, left: Arc<AtomicBool>, right: Arc<AtomicBool>) {
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
    if let Err(e) = run() {
        eprintln!("tobii-gaze-keys: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
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
            "--left" => left_edge = Some(args.next().ok_or("--left needs a value")?.parse()?),
            "--right" => right_edge = Some(args.next().ok_or("--right needs a value")?.parse()?),
            "--margin" => margin = args.next().ok_or("--margin needs a value")?.parse()?,
            "--no-super" => require_super = false,
            "--interval-ms" => {
                let ms: u64 = args.next().ok_or("--interval-ms needs a value")?.parse()?;
                interval = Duration::from_millis(ms);
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
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    let auto = left_edge.is_none() && right_edge.is_none();
    if !auto {
        // Partial manual spec: fill the unset side with a sane default.
        let l = left_edge.unwrap_or(0.15);
        let r = right_edge.unwrap_or(0.85);
        if !(l < r) {
            return Err("--left must be less than --right".into());
        }
        left_edge = Some(l);
        right_edge = Some(r);
    }
    if !(margin > 0.0 && margin < 0.5) {
        return Err("--margin must be between 0 and 0.5".into());
    }

    let kb = uinput::Keyboard::new(&[uinput::KEY_LEFT, uinput::KEY_RIGHT])
        .map_err(|e| format!("opening /dev/uinput ({e}); try running with sudo"))?;

    // Watch the physical Super/Meta key (unless gating is disabled).
    let supers = if require_super {
        Some(superkey::Watcher::spawn().map_err(|e| {
            format!("watching modifier keys ({e}); need read access to /dev/input/event*")
        })?)
    } else {
        None
    };

    let mut stream = tobii::ipc::connect_or_spawn()?;
    write_frame(&mut stream, &encode_subscribe(STREAM_GAZE))?;

    let gate = if require_super { " while Super held" } else { "" };
    if auto {
        println!(
            "tobii-gaze-keys: AUTO calibration (margin {margin:.2}), 1 tap / {}ms{gate}\n\
             look fully LEFT and fully RIGHT once to learn your gaze range...",
            interval.as_millis()
        );
    } else {
        println!(
            "tobii-gaze-keys: fixed edges <{:.2} / >{:.2}, 1 tap / {}ms{gate}",
            left_edge.unwrap(),
            right_edge.unwrap(),
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
        let Some(body) = read_frame(&mut stream)? else {
            return Err("daemon closed the connection".into());
        };
        match decode_server(&body) {
            Some(ServerMsg::Subscribed { ok: false }) => {
                return Err("daemon is busy with the other mode (head)".into());
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
                let (lo, hi, calibrating) = if let (Some(l), Some(r)) = (left_edge, right_edge) {
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
                let super_held = supers.as_ref().map_or(true, |s| s.down());

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
                    if last_left.map_or(true, |t| now.duration_since(t) >= interval) {
                        kb.tap(uinput::KEY_LEFT)?;
                        last_left = Some(now);
                        eprint!("\r<- LEFT  (x={x:.2})        ");
                    }
                } else if x >= hi {
                    last_left = None;
                    if last_right.map_or(true, |t| now.duration_since(t) >= interval) {
                        kb.tap(uinput::KEY_RIGHT)?;
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
            _ => {}
        }
    }
}
