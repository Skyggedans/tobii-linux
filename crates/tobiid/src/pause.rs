//! Device pause and resume: command 3100 with `u32 1` or `u32 0`, as the
//! DLL's `tracker_pause_device` / `tracker_resume_device` send it. Only the
//! resume was ever captured (every init replay sends it); the pause comes
//! from the DLL.
//!
//! The pause is one state for the device, as in the DLL: the last request
//! wins and any client may resume. Unlike the DLL, which learns the state
//! only from the device, a pause or resume the device accepts is reported at
//! once (state [`state::DEVICE_PAUSED`](tobii_ipc::request::state) and a
//! `DEVICE_PAUSED_STATE_CHANGED` notification when it changes). The device's
//! own notification (3110, never captured) is only logged until the hardware
//! shows what it carries.
//!
//! A pause ends without a request in three ways. The client that paused
//! last goes away: the device is resumed for it. The device re-initialises
//! (the init replay resumes it): after the engine is lost, and at every
//! `DeviceReady`. A resume never gets stuck: one the device does not answer
//! still counts, since the engine, no longer told the device is paused,
//! re-opens a device that stays silent, and the init replay resumes it.
//!
//! A pause and a calibration session exclude each other: pausing during a
//! session is `CALIBRATION_BUSY`, and a session cannot start while the device
//! is paused or a pause is on its way (see [`crate::calibration`]).
//!
//! Requests hold `pause_lock` from before the command to the state update,
//! so the device and `paused` agree; the state lock is taken only briefly
//! and never across the command.

use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::thread;
use std::time::{Duration, Instant};

use tobii_ipc::request::status;
use tobii_ipc::{
    Notification, NotificationValue, STREAM_NOTIFICATIONS, encode_notification, notification,
};
use tobii_proto::facts::device_pause_payload;
use tobii_proto::protocol::cmd;
use tracing::{info, warn};

use crate::daemon::{State, lock_state};
use crate::device::run;
use crate::requests::Reply;

/// How long the device may take to answer 3100. The DLL allows 3 s; the
/// init's resume has taken 3.5 s on Linux.
const PAUSE_TIMEOUT: Duration = Duration::from_secs(6);
/// How long a request waits for another pause or resume to finish.
#[cfg(not(test))]
const LOCK_TIMEOUT: Duration = Duration::from_secs(20);
#[cfg(test)]
const LOCK_TIMEOUT: Duration = Duration::from_millis(300);
/// How often a waiting request tries the lock.
const LOCK_POLL: Duration = Duration::from_millis(10);

/// Answer a pause (`[1]`) or resume (`[0]`) from `client`.
pub(crate) fn handle(state: &Mutex<State>, client: u64, payload: &[u8]) -> Reply {
    match payload {
        [1] => pause(state, client),
        [0] => resume(state, client),
        _ => Reply::err(status::INVALID_PARAMETER),
    }
}

/// `lock`, waiting until `deadline` at most.
fn lock_until(lock: &Mutex<()>, deadline: Instant) -> Option<MutexGuard<'_, ()>> {
    loop {
        match lock.try_lock() {
            Ok(guard) => return Some(guard),
            // It guards no data.
            Err(TryLockError::Poisoned(p)) => return Some(p.into_inner()),
            Err(TryLockError::WouldBlock) => {}
        }
        if Instant::now() >= deadline {
            return None;
        }
        thread::sleep(LOCK_POLL);
    }
}

/// The pause lock, as the state holds it.
fn pause_lock(state: &Mutex<State>) -> Arc<Mutex<()>> {
    Arc::clone(&lock_state(state).pause_lock)
}

fn pause(state: &Mutex<State>, client: u64) -> Reply {
    let deadline = Instant::now() + LOCK_TIMEOUT;
    let lock = pause_lock(state);
    let Some(_held) = lock_until(&lock, deadline) else {
        return Reply::err(status::TIMED_OUT);
    };
    let (device, losses) = {
        let mut st = lock_state(state);
        if st.calibration.is_active() {
            return Reply::err(status::CALIBRATION_BUSY);
        }
        // The engine takes commands only once its init is done, so the
        // pause always follows the init's resume.
        let Some(device) = st.device_for(client) else {
            return Reply::err(status::CONNECTION_FAILED);
        };
        st.pausing = true;
        st.set_pause_hint(true);
        (device, st.engine_losses)
    };
    let result = run(
        device.as_ref(),
        cmd::DEVICE_PAUSE,
        device_pause_payload(true),
        PAUSE_TIMEOUT,
    );
    let mut st = lock_state(state);
    st.pausing = false;
    let reply = match result {
        Ok(_) if st.engine_losses == losses => {
            info!(client, "device paused");
            record(&mut st, true, Some(client));
            Reply::ok(Vec::new())
        }
        Ok(_) => {
            warn!(
                client,
                "the engine was lost as the device paused; the next init resumes it"
            );
            Reply::err(status::CONNECTION_FAILED)
        }
        Err(code) => Reply::err(code),
    };
    let paused = st.paused;
    st.set_pause_hint(paused);
    reply
}

/// A resume the device does not answer still counts (see the module doc).
fn resume(state: &Mutex<State>, client: u64) -> Reply {
    let deadline = Instant::now() + LOCK_TIMEOUT;
    let lock = pause_lock(state);
    let Some(_held) = lock_until(&lock, deadline) else {
        return Reply::err(status::TIMED_OUT);
    };
    let device = lock_state(state).commands();
    let Some(device) = device else {
        // No engine: the next init resumes the device.
        let mut st = lock_state(state);
        record(&mut st, false, None);
        st.set_pause_hint(false);
        return Reply::ok(Vec::new());
    };
    let result = run(
        device.as_ref(),
        cmd::DEVICE_PAUSE,
        device_pause_payload(false),
        PAUSE_TIMEOUT,
    );
    let mut st = lock_state(state);
    let reply = match result {
        Ok(_) => Reply::ok(Vec::new()),
        Err(code @ (status::TIMED_OUT | status::CONNECTION_FAILED)) => {
            warn!(
                status = code,
                "the device did not answer the resume; it is re-opened if it stays silent"
            );
            Reply::ok(Vec::new())
        }
        Err(code) => Reply::err(code),
    };
    if reply.status == status::OK {
        info!(client, "device resumed");
        record(&mut st, false, None);
    }
    let paused = st.paused;
    st.set_pause_hint(paused);
    reply
}

/// `client` went away: if its pause is in effect, resume the device. Keyed
/// on the holder rather than the client list, so it works after the pump has
/// dropped the client too. Only the running engine is asked; none is
/// started for this.
pub(crate) fn release(state: &Mutex<State>, client: u64) {
    // The client's own requests ran on the thread calling this, so the
    // holder cannot become `client` after this check.
    if lock_state(state).pause_holder != Some(client) {
        return;
    }
    let deadline = Instant::now() + LOCK_TIMEOUT;
    let lock = pause_lock(state);
    let Some(_held) = lock_until(&lock, deadline) else {
        warn!(
            client,
            "pausing client disconnected; could not resume the device in time"
        );
        let mut st = lock_state(state);
        if st.pause_holder == Some(client) {
            record(&mut st, false, None);
            st.set_pause_hint(false);
        }
        return;
    };
    let device = {
        let mut st = lock_state(state);
        if st.pause_holder != Some(client) {
            return;
        }
        st.pause_holder = None;
        st.commands()
    };
    info!(client, "pausing client disconnected; resuming the device");
    if let Some(device) = device
        && let Err(code) = run(
            device.as_ref(),
            cmd::DEVICE_PAUSE,
            device_pause_payload(false),
            PAUSE_TIMEOUT,
        )
    {
        warn!(status = code, "could not resume the device");
    }
    let mut st = lock_state(state);
    if st.pause_holder.is_none() {
        record(&mut st, false, None);
    }
    let paused = st.paused;
    st.set_pause_hint(paused);
}

/// The device finished an init, whose replay resumed it.
pub(crate) fn on_device_ready(st: &mut State) {
    // For an engine that re-opened the device in place.
    st.set_pause_hint(false);
    record(st, false, None);
}

/// The engine went away; the next one's init resumes the device.
pub(crate) fn on_engine_lost(st: &mut State) {
    st.clock = None;
    st.engine_losses = st.engine_losses.wrapping_add(1);
    record(st, false, None);
}

/// The device's own pause notification (3110): logged only. It was never
/// captured, and one sent before an init would arrive with the init's other
/// messages after `DeviceReady`.
pub(crate) fn on_notification(paused: bool) {
    info!(paused, "device pause notification (3110); not acted on");
}

/// Note whether the device is paused and by whom, telling notification
/// subscribers when that changes.
fn record(st: &mut State, paused: bool, holder: Option<u64>) {
    st.pause_holder = holder;
    if st.paused != paused {
        st.paused = paused;
        let body = encode_notification(&Notification {
            kind: notification::DEVICE_PAUSED_STATE_CHANGED,
            value: NotificationValue::State(paused),
        });
        st.broadcast(STREAM_NOTIFICATIONS, &body);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::tests::{outbox, state_with_client};
    use crate::device::DeviceCommands;
    use tobii_calib::store::Location;
    use tobii_ipc::request::{Request, encode_u32, kind, state};
    use tobii_ipc::{ServerMsg, decode_server};
    use tobii_proto::facts::DeviceFacts;
    use tobii_usb::engine::{CommandError, CommandResponse, GazeSample, Sample};

    /// Records every command and answers each with `answer`.
    struct Recording {
        sent: Mutex<Vec<(u32, Vec<u8>)>>,
        answer: Mutex<Result<CommandResponse, CommandError>>,
    }

    impl DeviceCommands for Recording {
        fn run(
            &self,
            cmd: u32,
            payload: Vec<u8>,
            _timeout: Duration,
        ) -> Result<CommandResponse, CommandError> {
            self.sent.lock().expect("sent").push((cmd, payload));
            self.answer.lock().expect("answer").clone()
        }
    }

    struct Setup {
        state: Mutex<State>,
        device: Arc<Recording>,
    }

    /// Client 1 subscribes to notifications, client 2 does not; a stand-in
    /// device answers, so no engine starts.
    fn setup() -> Setup {
        let device = Arc::new(Recording {
            sent: Mutex::new(Vec::new()),
            answer: Mutex::new(Ok(CommandResponse::ok(Vec::new()))),
        });
        let mut st = state_with_client(1);
        st.clients.append(&mut state_with_client(2).clients);
        st.clients[0].streams = STREAM_NOTIFICATIONS;
        st.fake_device = Some(device.clone());
        Setup {
            state: Mutex::new(st),
            device,
        }
    }

    fn answer(s: &Setup, answer: Result<CommandResponse, CommandError>) {
        *s.device.answer.lock().expect("answer") = answer;
    }

    /// A request from `client`, through the daemon's dispatcher.
    fn ask(s: &Setup, client: u64, k: u8, payload: &[u8]) -> Reply {
        crate::requests::handle(
            &s.state,
            client,
            &Request {
                id: 1,
                kind: k,
                payload,
            },
        )
    }

    fn pause_as(s: &Setup, client: u64) -> Reply {
        ask(s, client, kind::DEVICE_PAUSE, &[1])
    }

    fn resume_as(s: &Setup, client: u64) -> Reply {
        ask(s, client, kind::DEVICE_PAUSE, &[0])
    }

    /// What STATE 2 answers.
    fn is_paused(s: &Setup) -> Reply {
        ask(s, 1, kind::STATE, &encode_u32(state::DEVICE_PAUSED))
    }

    /// The pause values of every paused-state notification queued for
    /// `client`.
    fn notified(s: &Setup, client: u64) -> Vec<bool> {
        outbox(&lock_state(&s.state), client)
            .iter()
            .filter_map(|body| match decode_server(body) {
                Some(ServerMsg::Notification(Notification {
                    kind: notification::DEVICE_PAUSED_STATE_CHANGED,
                    value: NotificationValue::State(paused),
                })) => Some(paused),
                _ => None,
            })
            .collect()
    }

    fn sent(s: &Setup) -> Vec<(u32, Vec<u8>)> {
        s.device.sent.lock().expect("sent").clone()
    }

    fn hint(s: &Setup) -> Option<bool> {
        lock_state(&s.state).pause_hint
    }

    /// Pause as client 2, and forget what that sent.
    fn paused_by_2() -> Setup {
        let s = setup();
        assert_eq!(pause_as(&s, 2), Reply::ok(Vec::new()));
        s.device.sent.lock().expect("sent").clear();
        s
    }

    #[test]
    fn a_pause_sends_3100_and_tells_notification_subscribers() {
        let s = setup();
        assert_eq!(is_paused(&s), Reply::ok(vec![0]));

        assert_eq!(pause_as(&s, 2), Reply::ok(Vec::new()));

        assert_eq!(sent(&s), [(cmd::DEVICE_PAUSE, device_pause_payload(true))]);
        assert_eq!(is_paused(&s), Reply::ok(vec![1]));
        assert_eq!(notified(&s, 1), [true]);
        assert!(
            outbox(&lock_state(&s.state), 2).is_empty(),
            "not a notification subscriber"
        );
        assert_eq!(hint(&s), Some(true));
        let st = lock_state(&s.state);
        assert_eq!(st.pause_holder, Some(2));
        assert!(!st.pausing);
        assert!(st.clients[1].holds_device, "the device stays up for it");
    }

    #[test]
    fn a_second_pause_is_sent_again_but_not_announced() {
        let s = paused_by_2();

        assert_eq!(pause_as(&s, 1), Reply::ok(Vec::new()));

        assert_eq!(sent(&s), [(cmd::DEVICE_PAUSE, device_pause_payload(true))]);
        assert_eq!(notified(&s, 1), [true]);
        assert_eq!(
            lock_state(&s.state).pause_holder,
            Some(1),
            "the last pause holds"
        );
    }

    #[test]
    fn any_client_may_resume() {
        let s = paused_by_2();

        assert_eq!(resume_as(&s, 1), Reply::ok(Vec::new()));

        assert_eq!(sent(&s), [(cmd::DEVICE_PAUSE, device_pause_payload(false))]);
        assert_eq!(is_paused(&s), Reply::ok(vec![0]));
        assert_eq!(notified(&s, 1), [true, false]);
        assert_eq!(hint(&s), Some(false));
        assert_eq!(lock_state(&s.state).pause_holder, None);
    }

    #[test]
    fn a_resume_without_an_engine_is_left_to_the_next_init() {
        let s = paused_by_2();
        // No engine and no stand-in: nothing may start one.
        lock_state(&s.state).fake_device = None;

        assert_eq!(resume_as(&s, 1), Reply::ok(Vec::new()));

        assert!(sent(&s).is_empty());
        assert_eq!(is_paused(&s), Reply::ok(vec![0]));
        assert_eq!(notified(&s, 1), [true, false]);
        assert!(lock_state(&s.state).engine.is_none());
    }

    #[test]
    fn a_refused_pause_changes_nothing() {
        let s = setup();
        answer(
            &s,
            Ok(CommandResponse {
                status: 2,
                ..CommandResponse::ok(Vec::new())
            }),
        );

        assert_eq!(pause_as(&s, 2), Reply::err(status::OPERATION_FAILED));

        assert_eq!(is_paused(&s), Reply::ok(vec![0]));
        assert!(notified(&s, 1).is_empty());
        assert_eq!(hint(&s), Some(false), "the hint goes back");
        let st = lock_state(&s.state);
        assert_eq!(st.pause_holder, None);
        assert!(!st.pausing);
    }

    #[test]
    fn a_resume_the_device_does_not_answer_still_counts() {
        for failure in [CommandError::Timeout, CommandError::EngineGone] {
            let s = paused_by_2();
            answer(&s, Err(failure));

            assert_eq!(resume_as(&s, 1), Reply::ok(Vec::new()));

            assert_eq!(is_paused(&s), Reply::ok(vec![0]));
            assert_eq!(hint(&s), Some(false));
        }
        // A refusal is an answer: still paused.
        let s = paused_by_2();
        answer(
            &s,
            Ok(CommandResponse {
                status: 2,
                ..CommandResponse::ok(Vec::new())
            }),
        );
        assert_eq!(resume_as(&s, 1), Reply::err(status::OPERATION_FAILED));
        assert_eq!(is_paused(&s), Reply::ok(vec![1]));
        assert_eq!(hint(&s), Some(true));
    }

    /// Answers every command, but the engine is lost before the answer is
    /// taken in.
    struct LostWhileAnswering(std::sync::Weak<Mutex<State>>);

    impl DeviceCommands for LostWhileAnswering {
        fn run(
            &self,
            _cmd: u32,
            _payload: Vec<u8>,
            _timeout: Duration,
        ) -> Result<CommandResponse, CommandError> {
            if let Some(state) = self.0.upgrade() {
                on_engine_lost(&mut lock_state(&state));
            }
            Ok(CommandResponse::ok(Vec::new()))
        }
    }

    #[test]
    fn a_pause_the_lost_engine_accepted_does_not_count() {
        let state = Arc::new_cyclic(|weak| {
            let mut st = state_with_client(1);
            st.clients[0].streams = STREAM_NOTIFICATIONS;
            st.fake_device = Some(Arc::new(LostWhileAnswering(weak.clone())));
            Mutex::new(st)
        });

        assert_eq!(
            handle(&state, 1, &[1]),
            Reply::err(status::CONNECTION_FAILED)
        );

        let st = lock_state(&state);
        assert!(!st.paused);
        assert_eq!(st.pause_holder, None);
        assert_eq!(st.pause_hint, Some(false));
        assert!(outbox(&st, 1).is_empty(), "nothing to announce");
    }

    #[test]
    fn a_pause_payload_is_one_or_zero() {
        let s = setup();
        for payload in [&[][..], &[2], &[1, 0]] {
            assert_eq!(
                ask(&s, 1, kind::DEVICE_PAUSE, payload),
                Reply::err(status::INVALID_PARAMETER),
                "{payload:?}"
            );
        }
        assert!(sent(&s).is_empty());
    }

    #[test]
    fn a_pause_waits_for_another_only_so_long() {
        let s = setup();
        let lock = pause_lock(&s.state);
        let _busy = lock.lock().expect("lock");

        assert_eq!(pause_as(&s, 1), Reply::err(status::TIMED_OUT));
        assert_eq!(resume_as(&s, 1), Reply::err(status::TIMED_OUT));

        assert!(sent(&s).is_empty());
        assert!(!lock_state(&s.state).pausing);
    }

    /// A stand-in setup whose calibrations are saved in a scratch directory
    /// (never the user's).
    fn calibration_setup(tag: &str) -> (Setup, std::path::PathBuf) {
        let s = setup();
        let dir = std::env::temp_dir().join(format!("tobiid-pause-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        lock_state(&s.state).calibration.location = Location::File(dir.join("calibration.bin"));
        (s, dir)
    }

    #[test]
    fn a_pause_and_a_calibration_session_exclude_each_other() {
        let (s, dir) = calibration_setup("session");
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::ok(Vec::new())
        );
        s.device.sent.lock().expect("sent").clear();

        assert_eq!(pause_as(&s, 2), Reply::err(status::CALIBRATION_BUSY));
        assert!(sent(&s).is_empty());
        assert_eq!(hint(&s), None, "never told");

        assert_eq!(ask(&s, 1, kind::CALIBRATION_STOP, &[1]).status, status::OK);
        assert_eq!(pause_as(&s, 2), Reply::ok(Vec::new()));
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::err(status::NOT_AVAILABLE)
        );

        assert_eq!(resume_as(&s, 2), Reply::ok(Vec::new()));
        lock_state(&s.state).pausing = true;
        assert_eq!(
            ask(&s, 1, kind::CALIBRATION_START, &[2]),
            Reply::err(status::NOT_AVAILABLE),
            "a pause on its way"
        );
        assert!(!lock_state(&s.state).calibration.is_active());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_holder_going_away_resumes_the_device() {
        let s = paused_by_2();

        release(&s.state, 1);
        assert!(sent(&s).is_empty(), "not the holder");
        assert_eq!(is_paused(&s), Reply::ok(vec![1]));

        release(&s.state, 2);
        assert_eq!(sent(&s), [(cmd::DEVICE_PAUSE, device_pause_payload(false))]);
        assert_eq!(is_paused(&s), Reply::ok(vec![0]));
        assert_eq!(notified(&s, 1), [true, false]);
        assert_eq!(hint(&s), Some(false));
    }

    #[test]
    fn the_holder_is_resumed_for_after_the_pump_dropped_it() {
        let s = paused_by_2();
        lock_state(&s.state).clients.retain(|c| c.id != 2);

        release(&s.state, 2);

        assert_eq!(sent(&s), [(cmd::DEVICE_PAUSE, device_pause_payload(false))]);
        assert_eq!(is_paused(&s), Reply::ok(vec![0]));
    }

    #[test]
    fn a_device_init_ends_a_pause() {
        let s = paused_by_2();

        lock_state(&s.state).observe(&Sample::DeviceReady(Arc::new(DeviceFacts::default())));

        assert_eq!(is_paused(&s), Reply::ok(vec![0]));
        assert_eq!(notified(&s, 1), [true, false]);
        assert_eq!(hint(&s), Some(false));
        assert_eq!(lock_state(&s.state).pause_holder, None);
        release(&s.state, 2);
        assert!(sent(&s).is_empty(), "nothing left to resume");
    }

    #[test]
    fn losing_the_engine_ends_a_pause() {
        let s = paused_by_2();
        lock_state(&s.state).clock = Some((5_000_000, 1));

        on_engine_lost(&mut lock_state(&s.state));

        assert_eq!(is_paused(&s), Reply::ok(vec![0]));
        assert_eq!(notified(&s, 1), [true, false]);
        let st = lock_state(&s.state);
        assert_eq!((st.pause_holder, st.clock), (None, None));
    }

    #[test]
    fn gaze_while_paused_does_not_date_the_clock() {
        let s = paused_by_2();
        let mut st = lock_state(&s.state);
        st.observe(&Sample::Gaze(Box::new(GazeSample::new(
            tobii_proto::gaze83::GazeFrame::default(),
            7,
        ))));
        assert_eq!((st.clock, st.gaze_frames), (None, 0));
    }

    #[test]
    fn timesync_is_not_available_while_paused() {
        let s = paused_by_2();
        assert_eq!(
            ask(&s, 1, kind::TIMESYNC, &[]),
            Reply::err(status::NOT_AVAILABLE)
        );
    }

    #[test]
    fn the_devices_own_pause_notification_is_only_logged() {
        let s = setup();
        lock_state(&s.state).observe(&Sample::Notification(
            tobii_proto::facts::DeviceNotification::DevicePausedChanged(true),
        ));
        assert_eq!(is_paused(&s), Reply::ok(vec![0]));
        assert!(notified(&s, 1).is_empty());
    }
}
