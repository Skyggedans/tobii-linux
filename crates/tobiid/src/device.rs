//! The device commands the daemon runs on behalf of clients, behind a trait so
//! the request handlers can be tested without a tracker.

use std::time::Duration;

use tobii_ipc::request::status;
use tobii_proto::protocol::{RESPONSE_STATUS_OK, ttp_error};
use tobii_usb::engine::{CommandError, CommandResponse, Commands, DeviceCommand};

/// Runs one device command and waits for the answer.
pub(crate) trait DeviceCommands: Send + Sync {
    /// Send `cmd` with `payload` and wait up to `timeout` for the response.
    fn run(
        &self,
        cmd: u32,
        payload: Vec<u8>,
        timeout: Duration,
    ) -> Result<CommandResponse, CommandError>;
}

impl DeviceCommands for Commands {
    fn run(
        &self,
        cmd: u32,
        payload: Vec<u8>,
        timeout: Duration,
    ) -> Result<CommandResponse, CommandError> {
        Commands::run(
            self,
            DeviceCommand {
                cmd,
                payload,
                timeout,
            },
        )
    }
}

/// Run a command and reduce the outcome to its payload or a Stream Engine
/// status code. An answer succeeds only with the OK status word and no TTP
/// error; anything else is the device refusing the command.
pub(crate) fn run(
    device: &dyn DeviceCommands,
    cmd: u32,
    payload: Vec<u8>,
    timeout: Duration,
) -> Result<Vec<u8>, u8> {
    run_with(device, cmd, payload, timeout, false)
}

/// [`run`] for a request that needs the device's calibration session
/// (collect, discard, clear, compute): refused for a bad state, it is
/// `CALIBRATION_NOT_STARTED`, the device having no session.
pub(crate) fn run_in_session(
    device: &dyn DeviceCommands,
    cmd: u32,
    payload: Vec<u8>,
    timeout: Duration,
) -> Result<Vec<u8>, u8> {
    run_with(device, cmd, payload, timeout, true)
}

/// [`run`] and [`run_in_session`]: a refusal maps through [`refusal`] with
/// `in_session`.
fn run_with(
    device: &dyn DeviceCommands,
    cmd: u32,
    payload: Vec<u8>,
    timeout: Duration,
    in_session: bool,
) -> Result<Vec<u8>, u8> {
    match device.run(cmd, payload, timeout) {
        Ok(r) if r.status == RESPONSE_STATUS_OK && r.error == ttp_error::NONE => Ok(r.payload),
        Ok(r) => {
            tracing::warn!(
                cmd,
                status = r.status,
                error = format_args!("{:#x}", r.error),
                "device refused the command"
            );
            Err(refusal(r.error, in_session))
        }
        Err(CommandError::Timeout) => Err(status::TIMED_OUT),
        Err(e) => {
            tracing::warn!(cmd, error = %e, "device command failed");
            Err(status::CONNECTION_FAILED)
        }
    }
}

/// The status a refusal carrying TTP `error` gives. It follows the Windows
/// engine's mapping for an invalid parameter and for a failed operation; the
/// codes the engine gives `INTERNAL` (not supported, unknown) come back as
/// `OPERATION_FAILED`, since the daemon protocol has no `INTERNAL`. A bad state
/// means no session only to a request that needs one (`in_session`); anywhere
/// else (a start, a stop, a read or write of the calibration) it is a failed
/// operation. The request decides, not the command id: a start sends 1060 as
/// a clear does.
fn refusal(error: u32, in_session: bool) -> u8 {
    match error {
        ttp_error::INVALID_PARAMETER => status::INVALID_PARAMETER,
        ttp_error::BAD_STATE if in_session => status::CALIBRATION_NOT_STARTED,
        _ => status::OPERATION_FAILED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tobii_proto::calibration;
    use tobii_proto::protocol::cmd::{DEVICE_PAUSE, DISPLAY_AREA_SET};

    /// Answers every command with one fixed outcome.
    struct Answering(Result<CommandResponse, CommandError>);

    impl DeviceCommands for Answering {
        fn run(
            &self,
            _cmd: u32,
            _payload: Vec<u8>,
            _timeout: Duration,
        ) -> Result<CommandResponse, CommandError> {
            self.0.clone()
        }
    }

    /// A device answering with `status` and `error`.
    fn answer(status: u32, error: u32) -> Answering {
        Answering(Ok(CommandResponse {
            status,
            error,
            payload: vec![7],
        }))
    }

    fn outcome(cmd: u32, status: u32, error: u32) -> Result<Vec<u8>, u8> {
        run(
            &answer(status, error),
            cmd,
            Vec::new(),
            Duration::from_secs(1),
        )
    }

    fn outcome_in_session(cmd: u32, status: u32, error: u32) -> Result<Vec<u8>, u8> {
        run_in_session(
            &answer(status, error),
            cmd,
            Vec::new(),
            Duration::from_secs(1),
        )
    }

    #[test]
    fn only_an_ok_status_without_an_error_succeeds() {
        assert_eq!(outcome(DISPLAY_AREA_SET, 1, 0), Ok(vec![7]));
        assert_eq!(
            outcome_in_session(calibration::cmd::COLLECT_2D, 1, 0),
            Ok(vec![7])
        );
        assert_eq!(
            outcome(DISPLAY_AREA_SET, 2, 0),
            Err(status::OPERATION_FAILED)
        );
        assert_eq!(
            outcome(DISPLAY_AREA_SET, 1, 0x2000_0407),
            Err(status::OPERATION_FAILED)
        );
    }

    #[test]
    fn a_ttp_error_maps_to_a_stream_engine_status() {
        for (status_word, error, expected) in [
            (1, ttp_error::INVALID_PARAMETER, status::INVALID_PARAMETER),
            (1, 0x2000_0407, status::OPERATION_FAILED),
            // Not supported: INTERNAL in the Windows engine.
            (1, 0x2000_0500, status::OPERATION_FAILED),
            (2, ttp_error::NONE, status::OPERATION_FAILED),
        ] {
            assert_eq!(
                outcome(DISPLAY_AREA_SET, status_word, error),
                Err(expected),
                "status {status_word}, error {error:#x}"
            );
            // A request that needs the session maps these as any other does.
            assert_eq!(
                outcome_in_session(calibration::cmd::COLLECT_2D, status_word, error),
                Err(expected),
                "in a session: status {status_word}, error {error:#x}"
            );
        }
    }

    #[test]
    fn a_bad_state_means_no_session_only_to_a_request_that_needs_one() {
        assert_eq!(
            outcome_in_session(calibration::cmd::COLLECT_2D, 1, ttp_error::BAD_STATE),
            Err(status::CALIBRATION_NOT_STARTED)
        );
        // The command id does not decide: a collect run as any other command
        // is a failed operation too.
        for cmd in [
            calibration::cmd::START,
            calibration::cmd::STOP,
            calibration::cmd::COLLECT_2D,
            calibration::cmd::READ,
            calibration::cmd::WRITE,
            1000,
            DISPLAY_AREA_SET,
            DEVICE_PAUSE,
        ] {
            assert_eq!(
                outcome(cmd, 1, ttp_error::BAD_STATE),
                Err(status::OPERATION_FAILED),
                "cmd {cmd}"
            );
        }
    }

    #[test]
    fn no_answer_is_a_timeout_or_a_lost_connection() {
        let ask = |e: CommandError| {
            run(
                &Answering(Err(e)),
                DISPLAY_AREA_SET,
                Vec::new(),
                Duration::from_secs(1),
            )
        };
        assert_eq!(ask(CommandError::Timeout), Err(status::TIMED_OUT));
        assert_eq!(
            ask(CommandError::EngineGone),
            Err(status::CONNECTION_FAILED)
        );
    }
}
