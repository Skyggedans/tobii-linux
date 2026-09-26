//! The application's logger, the `tobii_custom_log_t` it hands
//! `tobii_api_create`.
//!
//! libtobii is a cdylib with its own copy of `tracing` and installs no
//! subscriber, so nothing outside it, a C host or a Rust one, sees its
//! `tracing` events: this logger is the only way its diagnostics leave the
//! library. They are libtobii's own, and few: ERROR when `field_of_use` is
//! refused, a device cannot connect to `tobiid` or reconnect,
//! `tobii_device_process_callbacks` reports a lost connection (once per loss;
//! a loss a request runs into goes unlogged if a reconnect mends it before a
//! process call, the request's status having said it), or `tobiid` sends a
//! reply that does not decode; INFO when a device connects or reconnects. A
//! failing call is not logged as such: its status says it. Each line goes to
//! `tracing` too, as the crate's other events do.
//!
//! The logger is called synchronously, on the thread inside the `tobii_*` call
//! that has something to say, and never from the reader thread: a [`Logger`]
//! holds the application's context pointer, so it is not `Send`, and the
//! [`SharedLogger`] a device keeps it in never reaches the reader. libtobii
//! takes no lock of its own around it, lets every lock the logging call took
//! go before it logs (a line logged from inside a callback still runs under
//! that callback's, see below), and does not serialise calls across threads,
//! so threads using one device, or several, may log at once, and their lines
//! may interleave: a loss one thread's process reports can come after the
//! reconnect another thread made since. It runs under the callback guard (see
//! [`crate::device::call`]): a call from inside it that a stream callback
//! could not make either is `TOBII_ERROR_CALLBACK_IN_PROGRESS`, as for a
//! callback. The guard is the logging thread's own, so another thread the
//! logger hands the line to may use the device meanwhile. A line logged by a
//! call made from inside a callback (a refused `field_of_use`) runs under
//! that callback's device locks, so a logger, like a callback, must not
//! block on another thread's call into any device. These are
//! libtobii's guarantees, not the DLL's.
//!
//! The 4.1 DLL, for comparison: its error lines go through one helper,
//! 0x18015e360(api, level, fmt, ...), which takes no lock and calls
//! `log_func` on whatever thread called it, with the API's `log_context`. It
//! has 810 direct call sites, plus 120 through two error-name helpers (which
//! call it at 0x180001266 and 0x18015754a with the level they are given,
//! always 0), all at ERROR save three at INFO: "Connected to platform
//! module" on each connect and reconnect (0x180153c19), and a firmware
//! upgrade in progress (0x18014467e, 0x1801588e3). Nearly every failing call
//! logs such a line, `tobii_device_process_callbacks` on every call that
//! returns `TOBII_ERROR_CONNECTION_FAILED` (0x180143c74), and a device logs
//! through the API it was created from. Many callers hold the device's API
//! mutex (dev+0x4e0) around it, a failing call's error line included (a
//! subscribe, 0x18015cea0; `tobii_get_device_info`, 0x18014327b;
//! `tobii_get_track_box`, 0x180142dd1; `tobii_device_reconnect`, 0x180143943
//! and 0x1801439fe; `tobii_calibration_retrieve`, 0x180147c57), and a
//! thread of the DLL's own reaches it too (0x18002b230, through
//! 0x180169550). The helper is not the only path:
//! thunks forward the lines of the DLL's own sub-libraries straight to
//! `log_func`, at DEBUG and TRACE while it enumerates devices (0x18015b070,
//! installed at 0x18015d5c2), at their own level 0..4 from its legacy TTP
//! layer (0x1801706c0), and at a level taken from the message (0x18015d930).
//! The DLL sets its callback flag while enumeration logs, not around the
//! helper's lines. Where libtobii differs, it does on purpose: its own
//! diagnostics rather than a line per failing call, a lost connection once
//! rather than at the host's frame rate, and always the guard, never a lock
//! of the logging call's, never a thread of its own. A device copies its
//! API's logger, so it keeps logging after `tobii_api_destroy`, where the
//! DLL's would log through the freed API.

use std::ffi::{CString, c_void};
use std::fmt;

use crate::device::call;
use crate::status::{Status, TOBII_ERROR_INVALID_PARAMETER};
use crate::types::{CustomLog, LogFn, LogLevel, TOBII_LOG_LEVEL_ERROR, TOBII_LOG_LEVEL_INFO};

/// The application's logger, copied from its `tobii_custom_log_t`. The raw
/// context pointer keeps it from being `Send`, so it can never reach the
/// reader thread.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Logger {
    func: LogFn,
    context: *mut c_void,
}

/// A device's copy of its API's logger, which every thread calling into the
/// device may call, several at once. Only the device holds one, never the
/// reader thread's end of its connection.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SharedLogger(pub(crate) Logger);

// SAFETY: `tobii_api_create`'s contract makes the logger sound to call on any
// thread that calls into the API or a device created from it, from several
// at once, until those are destroyed. libtobii calls it only from inside such
// a call (see `emit`), never from a thread of its own, and never reads
// through `context` itself: moving or sharing the copy moves two pointers.
unsafe impl Send for SharedLogger {}
// SAFETY: as for `Send`: a shared copy is only ever called, which the
// contract allows from several threads at once.
unsafe impl Sync for SharedLogger {}

impl Logger {
    /// The logger `custom_log` describes: none for null, where the DLL
    /// installs one that does nothing, and `TOBII_ERROR_INVALID_PARAMETER`
    /// for one without `log_func`, as in the DLL (0x180144ae2).
    ///
    /// # Safety
    /// `custom_log` must be null or valid for reading one
    /// `tobii_custom_log_t`.
    pub(crate) unsafe fn from_c(custom_log: *const CustomLog) -> Result<Option<Self>, Status> {
        // SAFETY: null or readable, per the caller; `CustomLog` is `Copy`.
        let Some(&CustomLog {
            log_context,
            log_func,
        }) = (unsafe { custom_log.as_ref() })
        else {
            return Ok(None);
        };
        let func = log_func.ok_or(TOBII_ERROR_INVALID_PARAMETER)?;
        Ok(Some(Self {
            func,
            context: log_context,
        }))
    }
}

/// The levels libtobii logs at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Level {
    /// Something failed, and the application may want to know why.
    Error,
    /// A device connected or reconnected.
    Info,
}

impl Level {
    /// The `tobii_log_level_t` the application is given.
    const fn c(self) -> LogLevel {
        match self {
            Self::Error => TOBII_LOG_LEVEL_ERROR,
            Self::Info => TOBII_LOG_LEVEL_INFO,
        }
    }
}

/// Log one line: to `tracing`, and to the application's logger if it gave
/// one, on this thread, under the callback guard. A NUL in the line, which
/// would end it early, becomes U+FFFD.
pub(crate) fn emit(logger: Option<Logger>, level: Level, args: fmt::Arguments<'_>) {
    match level {
        Level::Error => tracing::error!("{args}"),
        Level::Info => tracing::info!("{args}"),
    }
    let Some(logger) = logger else {
        return;
    };
    let Ok(text) = CString::new(args.to_string().replace('\0', "\u{fffd}")) else {
        return;
    };
    // SAFETY: `tobii_api_create`'s contract makes `func` sound to call with
    // `context`, any level and a NUL-terminated string valid for the call, on
    // any thread that calls into the API or a device created from it, from
    // several at once, until those are destroyed; this runs inside such a
    // call, and `text` outlives it.
    call(|| unsafe { (logger.func)(logger.context, level.c(), text.as_ptr()) });
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::device::{device_ref, in_callback};
    use crate::status::TOBII_ERROR_CALLBACK_IN_PROGRESS;
    use std::cell::{Cell, RefCell};
    use std::ffi::{CStr, c_char};
    use std::ptr;
    use std::sync::{Mutex, MutexGuard, PoisonError};
    use std::thread::{self, ThreadId};

    /// A logger calling `log_func` with `log_context`.
    pub(crate) fn logger(log_func: LogFn, log_context: *mut c_void) -> Option<Logger> {
        Some(Logger {
            func: log_func,
            context: log_context,
        })
    }

    /// The lines a [`record`] logger got: level, text and the thread it ran
    /// on.
    #[derive(Debug, Default)]
    pub(crate) struct Recorder(RefCell<Vec<(LogLevel, String, ThreadId)>>);

    impl Recorder {
        /// A `tobii_custom_log_t` that records into this.
        pub(crate) fn custom_log(&self) -> CustomLog {
            CustomLog {
                log_context: ptr::from_ref(self).cast_mut().cast(),
                log_func: Some(record),
            }
        }

        /// The same, as a device's logger.
        pub(crate) fn logger(&self) -> Option<Logger> {
            logger(record, ptr::from_ref(self).cast_mut().cast())
        }

        /// The levels and texts recorded, each checked to have been logged
        /// on this thread.
        pub(crate) fn lines(&self) -> Vec<(LogLevel, String)> {
            let here = thread::current().id();
            self.0
                .borrow()
                .iter()
                .map(|(level, text, thread)| {
                    assert_eq!(*thread, here, "{text:?} was logged on another thread");
                    (*level, text.clone())
                })
                .collect()
        }
    }

    unsafe extern "C" fn record(context: *mut c_void, level: LogLevel, text: *const c_char) {
        // SAFETY: the tests pass a live `Recorder` as the context, and
        // libtobii a NUL-terminated line valid for the call.
        let (recorder, text) = unsafe { (&*context.cast::<Recorder>(), CStr::from_ptr(text)) };
        recorder.0.borrow_mut().push((
            level,
            text.to_string_lossy().into_owned(),
            thread::current().id(),
        ));
    }

    /// The lines a [`record_shared`] logger got, level and text, in the
    /// order they came, from whichever threads logged them: for a device
    /// that several threads use, where a [`Recorder`] would refuse the
    /// lines that did not come from the test's own thread.
    #[derive(Debug, Default)]
    pub(crate) struct SyncRecorder(Mutex<Vec<(LogLevel, String)>>);

    impl SyncRecorder {
        /// A `tobii_custom_log_t` that records into this.
        pub(crate) fn custom_log(&self) -> CustomLog {
            CustomLog {
                log_context: ptr::from_ref(self).cast_mut().cast(),
                log_func: Some(record_shared),
            }
        }

        /// The same, as a device's logger.
        pub(crate) fn logger(&self) -> Option<Logger> {
            logger(record_shared, ptr::from_ref(self).cast_mut().cast())
        }

        /// The levels and texts recorded so far.
        pub(crate) fn lines(&self) -> Vec<(LogLevel, String)> {
            self.recorded().clone()
        }

        /// The lines, locked, whole even if the lock is poisoned: a panic
        /// cannot unwind out of a logger, and nothing else holds the lock
        /// but to push or clone them.
        fn recorded(&self) -> MutexGuard<'_, Vec<(LogLevel, String)>> {
            self.0.lock().unwrap_or_else(PoisonError::into_inner)
        }
    }

    unsafe extern "C" fn record_shared(context: *mut c_void, level: LogLevel, text: *const c_char) {
        // SAFETY: the tests pass a live `SyncRecorder` as the context, which
        // is `Sync`, and libtobii a NUL-terminated line valid for the call.
        let (recorder, text) = unsafe { (&*context.cast::<SyncRecorder>(), CStr::from_ptr(text)) };
        let line = (level, text.to_string_lossy().into_owned());
        recorder.recorded().push(line);
    }

    #[test]
    fn a_null_custom_log_is_no_logger_and_one_without_log_func_is_refused() {
        let recorder = Recorder::default();
        let custom_log = recorder.custom_log();
        let no_func = CustomLog {
            log_func: None,
            ..custom_log
        };
        let no_context = CustomLog {
            log_context: ptr::null_mut(),
            ..custom_log
        };
        // SAFETY: null, or live locals.
        unsafe {
            assert!(matches!(Logger::from_c(ptr::null()), Ok(None)));
            assert!(matches!(
                Logger::from_c(&raw const no_func),
                Err(TOBII_ERROR_INVALID_PARAMETER)
            ));
            let Ok(Some(l)) = Logger::from_c(&raw const no_context) else {
                panic!("a null log_context is the application's to give");
            };
            assert!(l.context.is_null());
            let Ok(Some(l)) = Logger::from_c(&raw const custom_log) else {
                panic!("a logger");
            };
            emit(Some(l), Level::Info, format_args!("copied"));
        }
        assert_eq!(
            recorder.lines(),
            [(TOBII_LOG_LEVEL_INFO, "copied".to_owned())]
        );
    }

    #[test]
    fn emit_hands_the_logger_its_context_level_and_text() {
        let recorder = Recorder::default();

        emit(recorder.logger(), Level::Error, format_args!("one {}", 1));
        emit(recorder.logger(), Level::Info, format_args!("two"));
        emit(recorder.logger(), Level::Error, format_args!("a\0b"));
        emit(None, Level::Error, format_args!("to nobody"));

        assert_eq!(
            recorder.lines(),
            [
                (TOBII_LOG_LEVEL_ERROR, "one 1".to_owned()),
                (TOBII_LOG_LEVEL_INFO, "two".to_owned()),
                (TOBII_LOG_LEVEL_ERROR, "a\u{fffd}b".to_owned()),
            ]
        );
    }

    unsafe extern "C" fn reenter(context: *mut c_void, _level: LogLevel, _text: *const c_char) {
        // SAFETY: the test passes a live `Cell<Status>` as the context.
        let seen = unsafe { &*context.cast::<Cell<Status>>() };
        // SAFETY: a null handle is never dereferenced; the guard answers first.
        seen.set(match unsafe { device_ref(ptr::null_mut()) } {
            Err(status) => status,
            Ok(_) => 0,
        });
    }

    /// The logger runs under the guard, and leaves it as it found it: set
    /// when a line is logged from inside a callback.
    #[test]
    fn the_logger_runs_under_the_callback_guard_and_leaves_it_as_it_was() {
        let seen = Cell::new(-1);
        let reentering = logger(reenter, ptr::from_ref(&seen).cast_mut().cast());

        emit(reentering, Level::Error, format_args!("x"));

        assert_eq!(seen.get(), TOBII_ERROR_CALLBACK_IN_PROGRESS);
        assert!(!in_callback());
        let mut still_inside = false;
        call(|| {
            emit(reentering, Level::Error, format_args!("from a callback"));
            still_inside = in_callback();
        });
        assert!(still_inside, "the guard outlives the line");
        assert!(!in_callback());
    }
}
