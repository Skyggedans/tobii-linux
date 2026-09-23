//! The device commands the daemon runs on behalf of clients, behind a trait so
//! the request handlers can be tested without a tracker.

use std::time::Duration;

use tobii_ipc::request::status;
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

/// The response status word every successful answer carries.
const RESPONSE_OK: u32 = 1;

/// Run a command and reduce the outcome to its payload or a Stream Engine
/// status code.
pub(crate) fn run(
    device: &dyn DeviceCommands,
    cmd: u32,
    payload: Vec<u8>,
    timeout: Duration,
) -> Result<Vec<u8>, u8> {
    match device.run(cmd, payload, timeout) {
        Ok(r) if r.status == RESPONSE_OK => Ok(r.payload),
        Ok(r) => {
            tracing::warn!(cmd, status = r.status, "device refused the command");
            Err(status::OPERATION_FAILED)
        }
        Err(CommandError::Timeout) => Err(status::TIMED_OUT),
        Err(e) => {
            tracing::warn!(cmd, error = %e, "device command failed");
            Err(status::CONNECTION_FAILED)
        }
    }
}
