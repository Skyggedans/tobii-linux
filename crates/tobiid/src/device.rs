//! The device commands the daemon runs on behalf of clients, behind a trait so
//! the request handlers can be tested without a tracker.

use std::time::Duration;

use tobii_ipc::request::status;
use tobii_proto::calibration;
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
    match device.run(cmd, payload, timeout) {
        Ok(r) if r.status == RESPONSE_STATUS_OK && r.error == ttp_error::NONE => Ok(r.payload),
        Ok(r) => {
            tracing::warn!(
                cmd,
                status = r.status,
                error = format_args!("{:#x}", r.error),
                "device refused the command"
            );
            Err(refusal(cmd, r.error))
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
/// means no session only to a calibration command.
fn refusal(cmd: u32, error: u32) -> u8 {
    match error {
        ttp_error::INVALID_PARAMETER => status::INVALID_PARAMETER,
        ttp_error::BAD_STATE if is_calibration(cmd) => status::CALIBRATION_NOT_STARTED,
        _ => status::OPERATION_FAILED,
    }
}

/// Whether `cmd` is one of the calibration commands (1010 start through
/// 1110 write).
fn is_calibration(cmd: u32) -> bool {
    (calibration::cmd::START..=calibration::cmd::WRITE).contains(&cmd)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn outcome(cmd: u32, status: u32, error: u32) -> Result<Vec<u8>, u8> {
        let device = Answering(Ok(CommandResponse {
            status,
            error,
            payload: vec![7],
        }));
        run(&device, cmd, Vec::new(), Duration::from_secs(1))
    }

    #[test]
    fn only_an_ok_status_without_an_error_succeeds() {
        assert_eq!(outcome(DISPLAY_AREA_SET, 1, 0), Ok(vec![7]));
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
        let collect = calibration::cmd::COLLECT_2D;
        assert_eq!(
            outcome(collect, 1, ttp_error::INVALID_PARAMETER),
            Err(status::INVALID_PARAMETER)
        );
        assert_eq!(
            outcome(DISPLAY_AREA_SET, 1, ttp_error::INVALID_PARAMETER),
            Err(status::INVALID_PARAMETER)
        );
        for cmd in [
            calibration::cmd::START,
            collect,
            calibration::cmd::DISCARD_2D,
            calibration::cmd::WRITE,
        ] {
            assert_eq!(
                outcome(cmd, 1, ttp_error::BAD_STATE),
                Err(status::CALIBRATION_NOT_STARTED),
                "cmd {cmd}"
            );
        }
        for cmd in [1000, DISPLAY_AREA_SET, DEVICE_PAUSE] {
            assert_eq!(
                outcome(cmd, 1, ttp_error::BAD_STATE),
                Err(status::OPERATION_FAILED),
                "cmd {cmd}"
            );
        }
        assert_eq!(
            outcome(collect, 1, 0x2000_0407),
            Err(status::OPERATION_FAILED)
        );
        // Not supported: INTERNAL in the Windows engine.
        assert_eq!(
            outcome(collect, 1, 0x2000_0500),
            Err(status::OPERATION_FAILED)
        );
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
