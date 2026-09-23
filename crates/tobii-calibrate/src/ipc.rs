//! The daemon connection: requests answered synchronously, gaze samples
//! forwarded as they arrive (the verification view draws them).

use std::os::unix::net::UnixStream;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use tobii_ipc::request::encode_request;
use tobii_ipc::{
    STREAM_GAZE, STREAM_NOTIFICATIONS, ServerMsg, decode_server, encode_subscribe, read_frame,
    write_frame,
};

/// A reply: status and payload.
pub(crate) type Reply = (u8, Vec<u8>);

/// Something to do with each gaze sample: `(x, y, valid)`.
pub(crate) type GazeSink = Box<dyn Fn(f32, f32, bool) + Send>;

/// A connected client.
pub(crate) struct Connection {
    stream: UnixStream,
    replies: Receiver<(u32, Reply)>,
    gaze_seen: Receiver<()>,
    next_id: u32,
}

impl Connection {
    /// Connect (spawning the daemon if needed), subscribe to gaze and
    /// notifications, and hand every gaze sample to `on_gaze`.
    pub(crate) fn open(on_gaze: GazeSink) -> Result<Self> {
        let mut stream = tobii_ipc::connect_or_spawn().context("connecting to tobiid")?;
        let mut reader = stream.try_clone()?;
        let (reply_tx, replies) = mpsc::channel();
        let (seen_tx, gaze_seen) = mpsc::channel();
        thread::spawn(move || {
            while let Ok(Some(body)) = read_frame(&mut reader) {
                match decode_server(&body) {
                    Some(ServerMsg::Reply {
                        request_id,
                        status,
                        payload,
                    }) => {
                        if reply_tx.send((request_id, (status, payload))).is_err() {
                            break;
                        }
                    }
                    Some(ServerMsg::Gaze { valid, xy, .. }) => {
                        on_gaze(xy[0], xy[1], valid);
                        let _ = seen_tx.send(());
                    }
                    Some(ServerMsg::Notification(n)) => {
                        tracing::debug!(notification = ?n, "device notification");
                    }
                    _ => {}
                }
            }
        });
        write_frame(
            &mut stream,
            &encode_subscribe(STREAM_GAZE | STREAM_NOTIFICATIONS),
        )?;
        Ok(Self {
            stream,
            replies,
            gaze_seen,
            next_id: 0,
        })
    }

    /// Block until the tracker streams (a cold start takes several seconds).
    pub(crate) fn wait_for_gaze(&self, timeout: Duration) -> Result<()> {
        match self.gaze_seen.recv_timeout(timeout) {
            Ok(()) => Ok(()),
            Err(RecvTimeoutError::Timeout) => bail!("no gaze from the tracker within {timeout:?}"),
            Err(RecvTimeoutError::Disconnected) => bail!("tobiid closed the connection"),
        }
    }

    /// Send a request and wait for its reply.
    pub(crate) fn request(&mut self, kind: u8, payload: &[u8], timeout: Duration) -> Result<Reply> {
        self.next_id = self.next_id.wrapping_add(1).max(1);
        let id = self.next_id;
        write_frame(&mut self.stream, &encode_request(id, kind, payload))
            .context("tobiid connection lost")?;
        let deadline = Instant::now() + timeout;
        loop {
            match self
                .replies
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            {
                Ok((reply_id, reply)) if reply_id == id => return Ok(reply),
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => bail!("tobiid did not answer within {timeout:?}"),
                Err(RecvTimeoutError::Disconnected) => bail!("tobiid closed the connection"),
            }
        }
    }
}
