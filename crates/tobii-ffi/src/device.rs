//! The API and device handles: a device is a connection to the `tobiid`
//! daemon with a reader thread feeding decoded messages into a channel, the
//! registered callbacks, and a synchronous request/reply helper.
//!
//! Invariant: every stored callback was registered through the matching
//! `tobii_*_subscribe` entry point, whose safety contract makes it sound to
//! invoke with the stored `user_data` until it is unsubscribed or the device
//! is destroyed.

use std::cell::Cell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::fmt;
use std::io;
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tobii_ipc::request::{DeviceInfo as DeviceInfoMsg, encode_request};
use tobii_ipc::{
    self, NotificationValue as WireValue, STREAM_EYE_POSITION, STREAM_GAZE, STREAM_GAZE_DATA,
    STREAM_GAZE_ORIGIN, STREAM_HEAD, STREAM_IMAGE, STREAM_NOTIFICATIONS, STREAM_PRESENCE,
    ServerMsg, decode_server, encode_subscribe, read_frame, write_frame,
};

use crate::status::{
    Status, TOBII_ERROR_ALREADY_SUBSCRIBED, TOBII_ERROR_CALLBACK_IN_PROGRESS,
    TOBII_ERROR_CONFLICTING_API_INSTANCES, TOBII_ERROR_CONNECTION_FAILED,
    TOBII_ERROR_INVALID_PARAMETER, TOBII_ERROR_NO_ERROR, TOBII_ERROR_NOT_SUBSCRIBED,
    TOBII_ERROR_TIMED_OUT,
};
use crate::types::{
    DisplayArea, EyePair, EyePairFn, FieldOfUse, FieldOfUseFn, GazeData, GazeDataEye, GazeDataFn,
    GazePoint, GazePointFn, HeadPose, HeadPoseFn, Image, ImageFn, Notification, NotificationValue,
    NotificationsFn, PresenceFn, PresenceStatus, TOBII_VALIDITY_INVALID, TOBII_VALIDITY_VALID,
    Validity, copy_c_string,
};

/// How long a subscription change waits for the daemon's acknowledgement.
const SUBSCRIBE_ACK_TIMEOUT: Duration = Duration::from_secs(2);

thread_local! {
    /// Set while this thread runs a user callback. The entry points the crate
    /// documentation lists refuse a call made from inside one, as the Stream
    /// Engine does: re-entering with the device being dispatched would alias
    /// its `&mut`, and destroying it would free it under the dispatch loop.
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

/// Whether the current thread is inside a user callback.
pub(crate) fn in_callback() -> bool {
    IN_CALLBACK.get()
}

/// Opaque API handle. Carries no state; it exists so the entry points keep the
/// Stream Engine signatures.
#[derive(Debug)]
pub struct Api {
    _private: u8,
}

impl Api {
    pub(crate) fn new() -> Self {
        Self { _private: 0 }
    }
}

/// Opens a connection to the daemon (a socket pair in tests).
pub(crate) type Connector = Box<dyn FnMut() -> io::Result<UnixStream> + Send>;

/// One daemon connection and the thread reading it.
struct Link {
    stream: UnixStream,
    rx: Receiver<ServerMsg>,
    reader: Option<JoinHandle<()>>,
}

impl Link {
    fn open(connect: &mut Connector) -> io::Result<Self> {
        let stream = connect()?;
        let reader_stream = stream.try_clone()?;
        let (tx, rx) = mpsc::channel();
        let reader = thread::Builder::new()
            .name("tobii-ffi-reader".into())
            .spawn(move || reader_loop(reader_stream, &tx))?;
        Ok(Self {
            stream,
            rx,
            reader: Some(reader),
        })
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        // Shutting down the socket makes the reader's blocking read return, so
        // the join below cannot hang; a shutdown error only means it is already
        // closed.
        let _ = self.stream.shutdown(Shutdown::Both);
        if let Some(h) = self.reader.take()
            && h.join().is_err()
        {
            tracing::warn!("ffi reader thread panicked");
        }
    }
}

/// Pump frames from the daemon into `tx` until EOF, a read error, or the
/// receiving `Device` going away.
fn reader_loop(mut stream: UnixStream, tx: &Sender<ServerMsg>) {
    while let Ok(Some(body)) = read_frame(&mut stream) {
        if let Some(msg) = decode_server(&body)
            && tx.send(msg).is_err()
        {
            break;
        }
    }
}

/// A registered callback and its user data.
pub(crate) type Slot<F> = Option<(F, *mut c_void)>;

/// Every callback a device can have registered.
#[derive(Default)]
pub(crate) struct Callbacks {
    pub(crate) head: Slot<HeadPoseFn>,
    pub(crate) gaze: Slot<GazePointFn>,
    pub(crate) presence: Slot<PresenceFn>,
    pub(crate) gaze_origin: Slot<EyePairFn>,
    pub(crate) eye_position: Slot<EyePairFn>,
    pub(crate) user_position_guide: Slot<EyePairFn>,
    pub(crate) gaze_data: Slot<GazeDataFn>,
    pub(crate) image: Slot<ImageFn>,
    pub(crate) notifications: Slot<NotificationsFn>,
    /// Registered but never called: the field of use cannot change.
    pub(crate) field_of_use: Slot<FieldOfUseFn>,
}

impl Callbacks {
    /// The daemon streams these callbacks need; the single source of truth
    /// for the subscription mask.
    pub(crate) fn mask(&self) -> u32 {
        let mut mask = 0;
        let mut need = |on: bool, bit: u32| {
            if on {
                mask |= bit;
            }
        };
        need(self.head.is_some(), STREAM_HEAD);
        need(self.gaze.is_some(), STREAM_GAZE);
        need(self.presence.is_some(), STREAM_PRESENCE);
        need(self.gaze_origin.is_some(), STREAM_GAZE_ORIGIN);
        need(
            self.eye_position.is_some() || self.user_position_guide.is_some(),
            STREAM_EYE_POSITION,
        );
        need(self.gaze_data.is_some(), STREAM_GAZE_DATA);
        need(self.image.is_some(), STREAM_IMAGE);
        need(self.notifications.is_some(), STREAM_NOTIFICATIONS);
        mask
    }
}

/// Opaque device handle.
pub struct Device {
    link: Link,
    connect: Connector,
    pending: VecDeque<ServerMsg>,
    pub(crate) callbacks: Callbacks,
    pub(crate) field_of_use: FieldOfUse,
    /// Address of the API handle this device was created from.
    pub(crate) api: usize,
    next_request_id: u32,
    /// Identity, fetched once.
    pub(crate) device_info: Option<DeviceInfoMsg>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("streams", &self.callbacks.mask())
            .field("pending", &self.pending.len())
            .field("field_of_use", &self.field_of_use)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Connect through `connect` and start the reader thread.
    pub(crate) fn new(
        mut connect: Connector,
        api: usize,
        field_of_use: FieldOfUse,
    ) -> io::Result<Self> {
        let link = Link::open(&mut connect)?;
        Ok(Self {
            link,
            connect,
            pending: VecDeque::new(),
            callbacks: Callbacks::default(),
            field_of_use,
            api,
            next_request_id: 0,
            device_info: None,
        })
    }

    /// Connect to the daemon, spawning it if needed.
    #[cfg(not(test))]
    pub(crate) fn connect_daemon(api: usize, field_of_use: FieldOfUse) -> io::Result<Self> {
        Self::new(Box::new(tobii_ipc::connect_or_spawn), api, field_of_use)
    }

    /// Unit tests never reach the real daemon, which would open the tracker:
    /// their devices talk to `tests::fake_daemon`, and a constructor that
    /// gets this far fails as if no daemon could be reached.
    #[cfg(test)]
    pub(crate) fn connect_daemon(_api: usize, _field_of_use: FieldOfUse) -> io::Result<Self> {
        Err(io::ErrorKind::NotConnected.into())
    }

    /// The daemon connection, for fire-and-forget frames.
    pub(crate) fn stream(&self) -> &UnixStream {
        &self.link.stream
    }

    /// Wait for the next message, keeping anything else in `pending`.
    fn recv(&self, timeout: Duration) -> Result<ServerMsg, Status> {
        match self.link.rx.recv_timeout(timeout) {
            Ok(msg) => Ok(msg),
            Err(RecvTimeoutError::Timeout) => Err(TOBII_ERROR_TIMED_OUT),
            Err(RecvTimeoutError::Disconnected) => Err(TOBII_ERROR_CONNECTION_FAILED),
        }
    }

    /// Resend the subscription mask and wait for the daemon's ack, queueing
    /// any samples that arrive meanwhile. Returns the ack's `ok` flag.
    fn resend_subscription(&mut self) -> Result<bool, Status> {
        if write_frame(
            &mut self.link.stream,
            &encode_subscribe(self.callbacks.mask()),
        )
        .is_err()
        {
            return Err(TOBII_ERROR_CONNECTION_FAILED);
        }
        let deadline = Instant::now() + SUBSCRIBE_ACK_TIMEOUT;
        loop {
            match self.recv(deadline.saturating_duration_since(Instant::now()))? {
                ServerMsg::Subscribed { ok } => return Ok(ok),
                other => self.pending.push_back(other),
            }
        }
    }

    /// Register `callback` in `slot` and subscribe its stream. The Stream
    /// Engine's rules: a missing callback is invalid, an occupied slot is
    /// already subscribed.
    pub(crate) fn subscribe<F: Copy>(
        &mut self,
        slot: fn(&mut Callbacks) -> &mut Slot<F>,
        callback: Option<F>,
        user_data: *mut c_void,
    ) -> Status {
        let Some(callback) = callback else {
            return TOBII_ERROR_INVALID_PARAMETER;
        };
        if slot(&mut self.callbacks).is_some() {
            return TOBII_ERROR_ALREADY_SUBSCRIBED;
        }
        let before = self.callbacks.mask();
        *slot(&mut self.callbacks) = Some((callback, user_data));
        if self.callbacks.mask() == before {
            return TOBII_ERROR_NO_ERROR;
        }
        let status = match self.resend_subscription() {
            Ok(true) => TOBII_ERROR_NO_ERROR,
            Ok(false) => TOBII_ERROR_CONFLICTING_API_INSTANCES,
            Err(status) => status,
        };
        if status != TOBII_ERROR_NO_ERROR {
            *slot(&mut self.callbacks) = None;
        }
        status
    }

    /// Drop the callback in `slot` and, if no other callback needs its
    /// stream, tell the daemon.
    pub(crate) fn unsubscribe<F: Copy>(
        &mut self,
        slot: fn(&mut Callbacks) -> &mut Slot<F>,
    ) -> Status {
        let before = self.callbacks.mask();
        if slot(&mut self.callbacks).take().is_none() {
            return TOBII_ERROR_NOT_SUBSCRIBED;
        }
        if self.callbacks.mask() == before {
            return TOBII_ERROR_NO_ERROR;
        }
        match self.resend_subscription() {
            Ok(_) => TOBII_ERROR_NO_ERROR,
            Err(status) => status,
        }
    }

    /// Send a request and wait up to `timeout` for its reply, queueing the
    /// samples that arrive first. A reply with a non-zero status is that
    /// status.
    pub(crate) fn request(
        &mut self,
        kind: u8,
        payload: &[u8],
        timeout: Duration,
    ) -> Result<Vec<u8>, Status> {
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        let id = self.next_request_id;
        if write_frame(&mut self.link.stream, &encode_request(id, kind, payload)).is_err() {
            return Err(TOBII_ERROR_CONNECTION_FAILED);
        }
        let deadline = Instant::now() + timeout;
        loop {
            match self.recv(deadline.saturating_duration_since(Instant::now()))? {
                ServerMsg::Reply {
                    request_id,
                    status,
                    payload,
                } if request_id == id => {
                    return if status == 0 {
                        Ok(payload)
                    } else {
                        Err(Status::from(status))
                    };
                }
                ServerMsg::Reply { request_id, .. } => {
                    tracing::debug!(request_id, "stale reply dropped");
                }
                other => self.pending.push_back(other),
            }
        }
    }

    /// Open a fresh connection and restore the subscriptions.
    pub(crate) fn reconnect(&mut self) -> Status {
        match Link::open(&mut self.connect) {
            Ok(link) => {
                self.link = link;
                self.pending.clear();
                if self.callbacks.mask() == 0 {
                    return TOBII_ERROR_NO_ERROR;
                }
                match self.resend_subscription() {
                    Ok(_) => TOBII_ERROR_NO_ERROR,
                    Err(status) => status,
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "could not reconnect to tobiid");
                TOBII_ERROR_CONNECTION_FAILED
            }
        }
    }

    /// Drop every queued sample.
    pub(crate) fn clear_buffers(&mut self) {
        while self.link.rx.try_recv().is_ok() {}
        self.pending.clear();
    }

    /// Whether a sample is queued, waiting up to `timeout` for one.
    pub(crate) fn wait(&mut self, timeout: Duration) -> bool {
        if !self.pending.is_empty() {
            return true;
        }
        match self.link.rx.recv_timeout(timeout) {
            Ok(msg) => {
                self.pending.push_back(msg);
                true
            }
            Err(_) => false,
        }
    }

    /// Deliver every queued sample to its callbacks, on this thread.
    pub(crate) fn process(&mut self) {
        while let Ok(msg) = self.link.rx.try_recv() {
            self.pending.push_back(msg);
        }
        while let Some(msg) = self.pending.pop_front() {
            self.dispatch(&msg);
        }
    }

    /// Deliver one daemon message to the matching callbacks, if any.
    fn dispatch(&self, msg: &ServerMsg) {
        let cb = &self.callbacks;
        match msg {
            ServerMsg::Head {
                ts_us,
                pos_mm,
                rot_rad,
            } => {
                let hp = HeadPose {
                    timestamp_us: *ts_us,
                    position_validity: TOBII_VALIDITY_VALID,
                    position_xyz: *pos_mm,
                    rotation_validity_xyz: [TOBII_VALIDITY_VALID; 3],
                    rotation_xyz: *rot_rad,
                };
                if let Some((f, ud)) = cb.head {
                    // SAFETY: registered through `tobii_head_pose_subscribe`
                    // (see the module invariant); `hp` outlives the call.
                    call(|| unsafe { f(&raw const hp, ud) });
                }
            }
            ServerMsg::Gaze {
                ts_us, valid, xy, ..
            } => {
                let gp = GazePoint {
                    timestamp_us: *ts_us,
                    validity: validity(*valid),
                    position_xy: *xy,
                };
                if let Some((f, ud)) = cb.gaze {
                    // SAFETY: registered through `tobii_gaze_point_subscribe`;
                    // `gp` outlives the call.
                    call(|| unsafe { f(&raw const gp, ud) });
                }
            }
            ServerMsg::Presence { ts_us, status } => {
                if let Some((f, ud)) = cb.presence {
                    // SAFETY: registered through `tobii_user_presence_subscribe`,
                    // callable with any status, timestamp and `ud`.
                    call(|| unsafe { f(PresenceStatus::from(*status), *ts_us, ud) });
                }
            }
            ServerMsg::GazeOrigin(pair) => {
                let c = eye_pair(pair);
                if let Some((f, ud)) = cb.gaze_origin {
                    // SAFETY: registered through `tobii_gaze_origin_subscribe`;
                    // `c` outlives the call.
                    call(|| unsafe { f(&raw const c, ud) });
                }
            }
            ServerMsg::EyePosition(pair) => {
                let c = eye_pair(pair);
                for (f, ud) in [cb.eye_position, cb.user_position_guide]
                    .into_iter()
                    .flatten()
                {
                    // SAFETY: registered through
                    // `tobii_eye_position_normalized_subscribe` or
                    // `tobii_user_position_guide_subscribe`; `c` outlives the call.
                    call(|| unsafe { f(&raw const c, ud) });
                }
            }
            ServerMsg::GazeData(data) => {
                let c = GazeData {
                    timestamp_tracker_us: data.timestamp_tracker_us,
                    timestamp_system_us: data.timestamp_system_us,
                    left: gaze_data_eye(&data.left),
                    right: gaze_data_eye(&data.right),
                };
                if let Some((f, ud)) = cb.gaze_data {
                    // SAFETY: registered through `tobii_gaze_data_subscribe`;
                    // `c` outlives the call.
                    call(|| unsafe { f(&raw const c, ud) });
                }
            }
            ServerMsg::Image(image) => {
                let (Ok(width), Ok(height)) =
                    (i32::try_from(image.width), i32::try_from(image.height))
                else {
                    return;
                };
                let c = Image {
                    timestamp_us: image.ts_us,
                    width,
                    padding_per_row: 0,
                    height,
                    bits_per_pixel: i32::from(image.bits_per_pixel),
                    data: image.pixels.as_ptr().cast(),
                };
                if let Some((f, ud)) = cb.image {
                    // SAFETY: registered through `tobii_image_subscribe`; `c`
                    // and the pixels it points to outlive the call.
                    call(|| unsafe { f(&raw const c, ud) });
                }
            }
            ServerMsg::Notification(n) => {
                let c = notification(n);
                if let Some((f, ud)) = cb.notifications {
                    // SAFETY: registered through `tobii_notifications_subscribe`;
                    // `c` outlives the call.
                    call(|| unsafe { f(&raw const c, ud) });
                }
            }
            // Acks and replies nobody waited for, and message kinds a newer
            // daemon may add: nothing to deliver.
            _ => {}
        }
    }
}

/// Run a user callback with the re-entry guard set.
fn call(f: impl FnOnce()) {
    IN_CALLBACK.set(true);
    f();
    IN_CALLBACK.set(false);
}

fn validity(valid: bool) -> Validity {
    if valid {
        TOBII_VALIDITY_VALID
    } else {
        TOBII_VALIDITY_INVALID
    }
}

fn eye_pair(p: &tobii_ipc::EyePair) -> EyePair {
    EyePair {
        timestamp_us: p.ts_us,
        left_validity: validity(p.left.valid),
        left_xyz: p.left.xyz,
        right_validity: validity(p.right.valid),
        right_xyz: p.right.xyz,
    }
}

fn gaze_data_eye(e: &tobii_ipc::GazeDataEye) -> GazeDataEye {
    GazeDataEye {
        gaze_origin_validity: validity(e.gaze_origin_valid),
        gaze_origin_from_eye_tracker_mm_xyz: e.gaze_origin_mm,
        gaze_origin_in_track_box_normalized_xyz: e.gaze_origin_in_track_box,
        gaze_point_validity: validity(e.gaze_point_valid),
        gaze_point_from_eye_tracker_mm_xyz: e.gaze_point_mm,
        gaze_point_on_display_normalized_xy: e.gaze_point_on_display,
        eyeball_center_validity: validity(e.eyeball_center_valid),
        eyeball_center_from_eye_tracker_mm_xyz: e.eyeball_center_mm,
        pupil_validity: validity(e.pupil_valid),
        pupil_diameter_mm: e.pupil_diameter_mm,
    }
}

/// A display area in the C layout.
#[allow(clippy::cast_possible_truncation)] // reason: the C ABI carries float
pub(crate) fn display_area(a: &tobii_ipc::geometry::DisplayArea) -> DisplayArea {
    let f = |v: [f64; 3]| v.map(|c| c as f32);
    DisplayArea {
        top_left_mm_xyz: f(a.top_left_mm),
        top_right_mm_xyz: f(a.top_right_mm),
        bottom_left_mm_xyz: f(a.bottom_left_mm),
    }
}

/// A wire notification in the 520-byte C layout.
fn notification(n: &tobii_ipc::Notification) -> Notification {
    let mut c = Notification {
        type_: u32::from(n.kind),
        value_type: crate::types::TOBII_NOTIFICATION_VALUE_TYPE_NONE,
        value: NotificationValue { string_: [0; 512] },
    };
    match &n.value {
        WireValue::Float(v) => {
            c.value_type = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_FLOAT;
            c.value.float_ = *v;
        }
        WireValue::State(v) => {
            c.value_type = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_STATE;
            c.value.state = u32::from(*v);
        }
        WireValue::DisplayArea(a) => {
            c.value_type = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_DISPLAY_AREA;
            c.value.display_area = display_area(a);
        }
        WireValue::Uint(v) => {
            c.value_type = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_UINT;
            c.value.uint_ = *v;
        }
        WireValue::EnabledEye(v) => {
            c.value_type = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_ENABLED_EYE;
            c.value.enabled_eye = u32::from(*v);
        }
        WireValue::String(s) => {
            c.value_type = crate::types::TOBII_NOTIFICATION_VALUE_TYPE_STRING;
            let mut buf = [0; 512];
            copy_c_string(&mut buf, s);
            c.value.string_ = buf;
        }
        _ => {}
    }
    c
}

/// Borrow a device handle from C.
///
/// Refuses every call made from inside a callback (as the Stream Engine
/// does), then a null handle.
///
/// # Safety
/// `device` must be null or a live handle from `tobii_device_create` that no
/// other thread uses during the borrow.
pub(crate) unsafe fn device_mut<'a>(device: *mut Device) -> Result<&'a mut Device, Status> {
    if in_callback() {
        return Err(TOBII_ERROR_CALLBACK_IN_PROGRESS);
    }
    // SAFETY: the caller guarantees `device` is null or live and unaliased.
    unsafe { device.as_mut() }.ok_or(TOBII_ERROR_INVALID_PARAMETER)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use tobii_ipc::{
        STREAM_GAZE_ORIGIN, encode_gaze, encode_gaze_origin, encode_reply, encode_subscribed,
    };

    /// A daemon stand-in: answers each client frame with whatever `handler`
    /// returns (whole frame bodies), on a socket pair.
    pub(crate) fn fake_daemon(
        handler: impl FnMut(&[u8]) -> Vec<Vec<u8>> + Send + 'static,
    ) -> Connector {
        let handler = std::sync::Arc::new(std::sync::Mutex::new(handler));
        Box::new(move || {
            let (client, mut server) = UnixStream::pair()?;
            let handler = std::sync::Arc::clone(&handler);
            thread::spawn(move || {
                while let Ok(Some(body)) = read_frame(&mut server) {
                    let replies = (handler.lock().expect("handler"))(&body);
                    for r in replies {
                        if write_frame(&mut server, &r).is_err() {
                            return;
                        }
                    }
                }
            });
            Ok(client)
        })
    }

    /// Acks every subscription and answers every request with `status`/`payload`.
    pub(crate) fn device_with(status: u8, payload: Vec<u8>) -> Device {
        let connect = fake_daemon(move |body| match body.first() {
            Some(&tobii_ipc::TAG_SUBSCRIBE) => vec![encode_subscribed(true)],
            Some(&tobii_ipc::TAG_REQUEST) => {
                let req = tobii_ipc::request::decode_request(body).expect("request");
                vec![encode_reply(req.id, status, &payload)]
            }
            _ => vec![],
        });
        Device::new(connect, 1, 1).expect("device")
    }

    #[test]
    fn a_request_gets_its_reply_and_keeps_samples_that_came_first() {
        let connect = fake_daemon(|body| {
            let req = tobii_ipc::request::decode_request(body).expect("request");
            vec![
                encode_gaze(1, true, [0.5, 0.5], [f32::NAN; 2]),
                encode_reply(req.id.wrapping_add(7), 0, b"stale"),
                encode_reply(req.id, 0, b"answer"),
            ]
        });
        let mut d = Device::new(connect, 1, 1).expect("device");

        let got = d.request(1, &[], Duration::from_secs(2));

        assert_eq!(got, Ok(b"answer".to_vec()));
        assert_eq!(d.pending.len(), 1, "the gaze sample waits for process()");
    }

    #[test]
    fn a_failed_reply_is_its_status_and_silence_times_out() {
        let mut d = device_with(15, vec![]);
        assert_eq!(d.request(0x10, &[2], Duration::from_secs(2)), Err(15));

        let mut quiet = Device::new(fake_daemon(|_| vec![]), 1, 1).expect("device");
        assert_eq!(
            quiet.request(1, &[], Duration::from_millis(50)),
            Err(TOBII_ERROR_TIMED_OUT)
        );
    }

    unsafe extern "C" fn count_pair(_p: *const EyePair, ud: *mut c_void) {
        // SAFETY: the tests pass `&raw mut u32` as `ud`.
        unsafe { *ud.cast::<u32>() += 1 };
    }

    #[test]
    fn subscription_rules_follow_the_stream_engine() {
        let mut d = device_with(0, vec![]);
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();

        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, None::<EyePairFn>, ud),
            TOBII_ERROR_INVALID_PARAMETER
        );
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            0
        );
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            TOBII_ERROR_ALREADY_SUBSCRIBED
        );
        assert_eq!(d.callbacks.mask(), STREAM_GAZE_ORIGIN);
        assert_eq!(d.unsubscribe(|c| &mut c.gaze_origin), 0);
        assert_eq!(
            d.unsubscribe(|c| &mut c.gaze_origin),
            TOBII_ERROR_NOT_SUBSCRIBED
        );
    }

    /// Eye position and the user position guide share one daemon stream: the
    /// bit stays while either is subscribed, and each sample reaches both.
    #[test]
    fn a_shared_stream_feeds_both_callbacks() {
        let mut d = device_with(0, vec![]);
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();
        assert_eq!(
            d.subscribe(|c| &mut c.eye_position, Some(count_pair as EyePairFn), ud),
            0
        );
        assert_eq!(
            d.subscribe(
                |c| &mut c.user_position_guide,
                Some(count_pair as EyePairFn),
                ud
            ),
            0
        );

        d.dispatch(&ServerMsg::EyePosition(tobii_ipc::EyePair::default()));

        assert_eq!(hits, 2);
        assert_eq!(d.unsubscribe(|c| &mut c.eye_position), 0);
        assert_eq!(d.callbacks.mask(), STREAM_EYE_POSITION);
    }

    #[test]
    fn samples_are_dispatched_in_the_c_layout() {
        let mut d = device_with(0, vec![]);
        let mut hits = 0u32;
        let ud = (&raw mut hits).cast::<c_void>();
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            0
        );
        let body = encode_gaze_origin(&tobii_ipc::EyePair::default());
        d.pending.push_back(decode_server(&body).expect("decodes"));

        d.process();

        assert_eq!(hits, 1);
    }

    unsafe extern "C" fn reenter(_p: *const EyePair, ud: *mut c_void) {
        // SAFETY: the test passes `&raw mut Status` as `ud`.
        let out = unsafe { &mut *ud.cast::<Status>() };
        // SAFETY: a null handle is never dereferenced; the guard answers first.
        *out = match unsafe { device_mut(std::ptr::null_mut()) } {
            Err(s) => s,
            Ok(_) => 0,
        };
    }

    #[test]
    fn calls_from_inside_a_callback_are_refused() {
        let mut d = device_with(0, vec![]);
        let mut seen: Status = -1;
        let ud = (&raw mut seen).cast::<c_void>();
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(reenter as EyePairFn), ud),
            0
        );

        d.dispatch(&ServerMsg::GazeOrigin(tobii_ipc::EyePair::default()));

        assert_eq!(seen, TOBII_ERROR_CALLBACK_IN_PROGRESS);
        assert!(!in_callback());
    }

    #[test]
    fn notifications_fill_the_520_byte_union() {
        let n = notification(&tobii_ipc::Notification {
            kind: tobii_ipc::notification::CALIBRATION_ID_CHANGED,
            value: WireValue::Uint(0x7186_ba7d),
        });
        assert_eq!(
            (n.type_, n.value_type),
            (8, crate::types::TOBII_NOTIFICATION_VALUE_TYPE_UINT)
        );
        // SAFETY: `value_type` says `uint_` is the active field.
        assert_eq!(unsafe { n.value.uint_ }, 0x7186_ba7d);

        let s = notification(&tobii_ipc::Notification {
            kind: 10,
            value: WireValue::String("x".repeat(600)),
        });
        // SAFETY: `value_type` says `string_` is the active field.
        let bytes = unsafe { s.value.string_ };
        assert_eq!(bytes[510], c_char_of(b'x'));
        assert_eq!(bytes[511], 0, "truncated and terminated");
    }

    fn c_char_of(b: u8) -> std::ffi::c_char {
        std::ffi::c_char::from_ne_bytes([b])
    }

    #[test]
    fn reconnect_restores_the_subscription() {
        let mut d = device_with(0, vec![]);
        let ud = std::ptr::null_mut();
        assert_eq!(
            d.subscribe(|c| &mut c.gaze_origin, Some(count_pair as EyePairFn), ud),
            0
        );
        assert_eq!(d.reconnect(), 0);
        assert_eq!(d.callbacks.mask(), STREAM_GAZE_ORIGIN);
        d.clear_buffers();
        assert!(!d.wait(Duration::from_millis(10)));
    }
}
