//! `ipc-probe`: talk to a running `tobiid` the way a client would, ask it
//! every question the protocol has, optionally set a display area, and report
//! what each stream delivers. The quickest end-to-end check of the daemon.

use std::collections::BTreeMap;
use std::os::unix::net::UnixStream;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use tobii_ipc::geometry::display_area_basic;
use tobii_ipc::request::{
    self, decode_device_info, decode_display_area, decode_geometry_mounting,
    decode_hardware_configuration, decode_stream_types, decode_timesync, decode_track_box,
    encode_display_area, encode_request, kind, state,
};
use tobii_ipc::{ServerMsg, decode_server, encode_subscribe, read_frame, write_frame};

/// How long the daemon may take to answer (a cold device start included).
const REPLY_TIMEOUT: Duration = Duration::from_secs(20);

struct Probe {
    stream: UnixStream,
    rx: mpsc::Receiver<ServerMsg>,
    next_id: u32,
    early: Vec<ServerMsg>,
}

impl Probe {
    fn connect() -> Result<Self> {
        let stream = tobii_ipc::connect_or_spawn().context("connecting to tobiid")?;
        let mut reader = stream.try_clone()?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            while let Ok(Some(body)) = read_frame(&mut reader) {
                if let Some(msg) = decode_server(&body)
                    && tx.send(msg).is_err()
                {
                    break;
                }
            }
        });
        Ok(Self {
            stream,
            rx,
            next_id: 0,
            early: Vec::new(),
        })
    }

    /// Send a request and wait for its reply; samples that arrive first are
    /// kept for the stream report.
    fn ask(&mut self, k: u8, payload: &[u8]) -> Result<(u8, Vec<u8>)> {
        self.next_id += 1;
        let id = self.next_id;
        write_frame(&mut self.stream, &encode_request(id, k, payload))?;
        let deadline = Instant::now() + REPLY_TIMEOUT;
        loop {
            match self
                .rx
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            {
                Ok(ServerMsg::Reply {
                    request_id,
                    status,
                    payload,
                }) if request_id == id => return Ok((status, payload)),
                Ok(other) => self.early.push(other),
                Err(_) => bail!("no reply to request kind {k} within {REPLY_TIMEOUT:?}"),
            }
        }
    }
}

fn show(name: &str, answer: &(u8, Vec<u8>), decoded: impl FnOnce(&[u8]) -> String) {
    match answer {
        (0, payload) => println!("{name:18} {}", decoded(payload)),
        (status, _) => println!("{name:18} status {status}"),
    }
}

fn kind_of(msg: &ServerMsg) -> &'static str {
    match msg {
        ServerMsg::Head { .. } => "head",
        ServerMsg::Gaze { .. } => "gaze",
        ServerMsg::Presence { .. } => "presence",
        ServerMsg::GazeOrigin(_) => "gaze_origin",
        ServerMsg::EyePosition(_) => "eye_position",
        ServerMsg::GazeData(_) => "gaze_data",
        ServerMsg::GazeRaw(_) => "gaze_raw",
        ServerMsg::Image(_) => "image",
        ServerMsg::Notification(_) => "notification",
        ServerMsg::Subscribed { .. } => "subscribed",
        _ => "other",
    }
}

/// Run the probe.
///
/// # Errors
/// Fails when the daemon cannot be reached or stops answering.
pub(crate) fn run(streams: u32, secs: u64, set_display: Option<(f64, f64, f64)>) -> Result<()> {
    let mut p = Probe::connect()?;
    write_frame(&mut p.stream, &encode_subscribe(streams))?;

    let info = p.ask(kind::DEVICE_INFO, &[])?;
    show("device info", &info, |b| {
        format!("{:?}", decode_device_info(b))
    });
    let track_box = p.ask(kind::TRACK_BOX, &[])?;
    show("track box", &track_box, |b| {
        decode_track_box(b).map_or("?".into(), |t| {
            format!("front {:?} back {:?}", t.corners_mm[0], t.corners_mm[4])
        })
    });
    let mounting = p.ask(kind::GEOMETRY_MOUNTING, &[])?;
    show("mounting", &mounting, |b| {
        format!("{:?}", decode_geometry_mounting(b))
    });
    let area = p.ask(kind::DISPLAY_AREA_GET, &[])?;
    show("display area", &area, |b| {
        decode_display_area(b).map_or("?".into(), |a| {
            format!(
                "{:.1} x {:.1} mm, TL {:?}",
                a.top_right_mm[0] - a.top_left_mm[0],
                (0..3)
                    .map(|i| (a.top_left_mm[i] - a.bottom_left_mm[i]).powi(2))
                    .sum::<f64>()
                    .sqrt(),
                a.top_left_mm
            )
        })
    });
    let catalogue = p.ask(kind::STREAM_TYPES, &[])?;
    show("stream types", &catalogue, |b| {
        decode_stream_types(b).map_or("?".into(), |types| {
            types
                .iter()
                .map(|t| format!("{:#x} {} ({})", t.id, t.name, t.value))
                .collect::<Vec<_>>()
                .join(", ")
        })
    });
    let hardware = p.ask(kind::HARDWARE_CONFIGURATION, &[])?;
    show("hardware config", &hardware, |b| {
        decode_hardware_configuration(b).map_or("?".into(), |h| {
            format!(
                "{} entries, points {:?}, mode {}",
                h.entries.len(),
                h.points_mm,
                h.mode
            )
        })
    });
    let name = p.ask(kind::DEVICE_NAME_GET, &[])?;
    show("device name", &name, |b| {
        format!("{:?}", String::from_utf8_lossy(b))
    });
    let id = p.ask(kind::STATE, &request::encode_u32(state::CALIBRATION_ID))?;
    show("calibration id", &id, |b| {
        format!("{:?}", request::decode_u32(b))
    });
    let active = p.ask(kind::STATE, &request::encode_u32(state::CALIBRATION_ACTIVE))?;
    show("calibrating", &active, |b| {
        format!("{}", b.first().is_some_and(|v| *v != 0))
    });
    let paused = p.ask(kind::STATE, &request::encode_u32(state::DEVICE_PAUSED))?;
    show("paused", &paused, |b| {
        format!("{}", b.first().is_some_and(|v| *v != 0))
    });
    let faults = p.ask(kind::STATE, &request::encode_u32(state::FAULT))?;
    show("faults", &faults, |b| {
        format!("{:?}", String::from_utf8_lossy(b))
    });
    let warnings = p.ask(kind::STATE, &request::encode_u32(state::WARNING))?;
    show("warnings", &warnings, |b| {
        format!("{:?}", String::from_utf8_lossy(b))
    });

    if let Some((w, h, x)) = set_display {
        let Some(m) = (mounting.0 == 0)
            .then(|| decode_geometry_mounting(&mounting.1))
            .flatten()
        else {
            bail!("the daemon did not report the mounting; cannot compute a display area");
        };
        let area = display_area_basic(w, h, x, &m);
        let reply = p.ask(kind::DISPLAY_AREA_SET, &encode_display_area(&area))?;
        show("set display", &reply, |_| {
            format!("{w} x {h} mm, offset {x}")
        });
        let area = p.ask(kind::DISPLAY_AREA_GET, &[])?;
        show("display area now", &area, |b| {
            format!("{:?}", decode_display_area(b))
        });
    }

    println!("watching streams {streams:#x} for {secs} s ...");
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut first: BTreeMap<&str, String> = BTreeMap::new();
    let mut record = |msg: ServerMsg| {
        let k = kind_of(&msg);
        *counts.entry(k).or_default() += 1;
        first.entry(k).or_insert_with(|| match &msg {
            ServerMsg::Image(i) => format!(
                "{}x{} bpp {} ts {}",
                i.width, i.height, i.bits_per_pixel, i.ts_us
            ),
            other => format!("{other:?}"),
        });
    };
    for msg in std::mem::take(&mut p.early) {
        record(msg);
    }
    let end = Instant::now() + Duration::from_secs(secs);
    while let Ok(msg) =
        p.rx.recv_timeout(end.saturating_duration_since(Instant::now()))
    {
        record(msg);
    }
    let sync = p.ask(kind::TIMESYNC, &[])?;
    show("timesync", &sync, |b| format!("{:?}", decode_timesync(b)));
    for (k, n) in &counts {
        #[allow(clippy::cast_precision_loss)] // reason: a rate for display
        let rate = *n as f64 / secs.max(1) as f64;
        println!(
            "{k:14} {n:6}  ({rate:.1}/s)  first: {}",
            first.get(k).map_or("", String::as_str)
        );
    }
    Ok(())
}
