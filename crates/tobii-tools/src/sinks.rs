//! File/stdout sinks for the replay/track CLI paths: the decoded-candidate
//! CSV and the JSON Lines frame stream, plus the
//! per-packet fan-out that feeds them together with the `OpenTrack` sender and
//! the terminal dashboard.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use tracing::info;

use crate::dashboard::{print_live_decoded, render_tracking_dashboard};
use crate::opentrack::OpentrackUdp;
use tobii_proto::decode::{
    DERIVED_FIELDS, LIVE_FIELDS, TrackingFrame, decode_stream_payload, derive_live_values,
};
use tobii_proto::gaze83::{GazeFrame, decode_gaze_frame};
use tobii_proto::protocol::{marker, parse_message};
use tobii_proto::time::now_us;

/// CSV of the decoded stream candidates, one row per 0x53 packet, to a file
/// (or, in tests, any writer).
#[derive(Debug)]
pub(crate) struct DecodedCsv<W: Write = File> {
    pub(crate) out: BufWriter<W>,
}

impl DecodedCsv {
    /// Create (truncate) the CSV at `path` and write the header row.
    ///
    /// # Errors
    /// Fails when the file cannot be created or the header cannot be written.
    pub(crate) fn create(path: &str) -> Result<Self> {
        let file =
            File::create(path).with_context(|| format!("failed to create decoded CSV {path}"))?;
        let csv = Self::with_header(file)?;

        info!(path, "logging decoded stream candidates");
        Ok(csv)
    }
}

impl<W: Write> DecodedCsv<W> {
    /// Start the CSV on `out` with its header row.
    ///
    /// # Errors
    /// Fails when the header cannot be written.
    fn with_header(out: W) -> Result<Self> {
        let mut out = BufWriter::new(out);

        write!(out, "ts_us,packet")?;
        for field in DERIVED_FIELDS {
            write!(out, ",{field}")?;
        }
        for field in LIVE_FIELDS {
            write!(out, ",{}", field.name)?;
        }
        writeln!(out)?;

        Ok(Self { out })
    }

    /// Append one row: timestamp, packet number, derived fields, live fields.
    /// `gaze` is the same packet decoded by key, for the pupil columns.
    ///
    /// # Errors
    /// Fails when the write fails.
    pub(crate) fn write_packet(
        &mut self,
        packet_no: u64,
        values: &BTreeMap<(u32, usize, usize), f64>,
        gaze: Option<&GazeFrame>,
    ) -> Result<()> {
        let derived = derive_live_values(values, gaze);
        write!(self.out, "{},{}", now_us(), packet_no)?;
        for value in derived {
            write_csv_value(&mut self.out, value)?;
        }
        for field in LIVE_FIELDS {
            write_csv_value(&mut self.out, values.get(&field.key()).copied())?;
        }
        writeln!(self.out)?;
        self.out.flush()?;
        Ok(())
    }
}

/// JSON Lines stream of [`TrackingFrame`]s, to a file or stdout (`-`).
pub(crate) struct JsonlOutput {
    pub(crate) out: BufWriter<Box<dyn Write>>,
}

impl JsonlOutput {
    /// Open the JSONL sink; `-` selects stdout.
    ///
    /// # Errors
    /// Fails when the file cannot be created.
    pub(crate) fn create(path: &str) -> Result<Self> {
        let writer: Box<dyn Write> = if path == "-" {
            info!(path = "stdout", "logging tracking frames as JSON Lines");
            Box::new(io::stdout())
        } else {
            info!(path, "logging tracking frames as JSON Lines");
            Box::new(
                File::create(path)
                    .with_context(|| format!("failed to create JSONL output {path}"))?,
            )
        };

        Ok(Self {
            out: BufWriter::new(writer),
        })
    }

    /// Append one frame as a single JSON line and flush.
    ///
    /// # Errors
    /// Fails when the write fails.
    pub(crate) fn write_frame(&mut self, frame: &TrackingFrame) -> Result<()> {
        frame.write_json(&mut self.out)?;
        writeln!(self.out)?;
        self.out.flush()?;
        Ok(())
    }
}

/// Write one `,value` CSV cell (6 decimals), or a bare `,` for `None`.
///
/// # Errors
/// Fails when the write fails.
pub(crate) fn write_csv_value<W: Write>(out: &mut W, value: Option<f64>) -> Result<()> {
    match value {
        Some(value) => write!(out, ",{value:.6}")?,
        None => write!(out, ",")?,
    }
    Ok(())
}

/// Fan one IN packet out to every enabled sink. Non-stream packets (marker
/// other than 0x53) and empty decodes are ignored.
///
/// # Errors
/// Fails when decoding fails or any sink write/send fails.
pub(crate) fn handle_live_decoded<W: Write>(
    packet_no: u64,
    data: &[u8],
    live_csv: &mut Option<DecodedCsv<W>>,
    jsonl: &mut Option<JsonlOutput>,
    opentrack: &mut Option<OpentrackUdp>,
    print_decoded: bool,
    dashboard: bool,
) -> Result<()> {
    if marker(data) != Some(0x53) {
        return Ok(());
    }

    let decoded = decode_stream_payload(data)?;
    if decoded.is_empty() {
        return Ok(());
    }

    // The same packet by key, for the pupils the occurrence map lacks.
    let gaze = parse_message(data).as_ref().and_then(decode_gaze_frame);
    let frame = TrackingFrame::from_decoded(packet_no, &decoded, gaze.as_ref());

    if let Some(csv) = live_csv {
        csv.write_packet(packet_no, &decoded, gaze.as_ref())?;
    }

    if let Some(jsonl) = jsonl {
        jsonl.write_frame(&frame)?;
    }

    let mut opentrack_pose = None;
    if let Some(opentrack) = opentrack {
        opentrack_pose = opentrack.send_frame(&frame, &decoded)?;
    }

    if print_decoded {
        print_live_decoded(&frame);
    }

    if dashboard {
        render_tracking_dashboard(&frame, opentrack_pose)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use tobii_proto::protocol::hex_to_bytes;

    /// A writer the test reads back after handing a clone to a sink.
    #[derive(Clone, Default)]
    struct Shared(Rc<RefCell<Vec<u8>>>);

    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.borrow_mut().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// The session1 gaze packet, as it comes off the stream.
    fn session1_packet() -> Vec<u8> {
        hex_to_bytes(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tobii-proto/fixtures/session1-gaze-frame.hex"
        )))
        .expect("fixture is valid hex")
    }

    /// A gaze packet's CSV row carries the pupil diameters (keys
    /// `0x06`/`0x0c`), which only the keyed decode of the packet has, under
    /// their own header, and the secondary range under its.
    #[test]
    fn a_gaze_packet_reaches_the_csv_with_its_pupils() {
        let written = Shared::default();
        let mut csv = Some(DecodedCsv::with_header(written.clone()).expect("header"));

        handle_live_decoded::<Shared>(
            1,
            &session1_packet(),
            &mut csv,
            &mut None,
            &mut None,
            false,
            false,
        )
        .expect("handled");

        let text = String::from_utf8(written.0.borrow().clone()).expect("utf-8");
        let lines: Vec<Vec<&str>> = text.lines().map(|line| line.split(',').collect()).collect();
        let [header, row] = lines.as_slice() else {
            panic!("expected a header and one row: {text}");
        };
        assert_eq!(header.len(), row.len(), "{text}");
        let cell = |name: &str| {
            header
                .iter()
                .position(|column| *column == name)
                .map(|at| row[at])
        };
        assert_eq!(cell("pupil_diameter_left"), Some("6.247360"));
        assert_eq!(cell("pupil_diameter_right"), Some("5.996613"));
        assert_eq!(cell("secondary_range_left"), Some("435.728485"));
        assert_eq!(cell("secondary_range_right"), Some("430.605499"));
    }

    /// A gaze packet's JSON line carries the pupil diameters (keys
    /// `0x06`/`0x0c`), which only the keyed decode of the packet has.
    #[test]
    fn a_gaze_packet_reaches_the_jsonl_with_its_pupils() {
        let written = Shared::default();
        let mut jsonl = Some(JsonlOutput {
            out: BufWriter::new(Box::new(written.clone())),
        });

        handle_live_decoded::<File>(
            1,
            &session1_packet(),
            &mut None,
            &mut jsonl,
            &mut None,
            false,
            false,
        )
        .expect("handled");

        let json = String::from_utf8(written.0.borrow().clone()).expect("utf-8");
        assert!(
            json.contains("\"pupil_diameter\":{\"left\":6.247360,\"right\":5.996613}"),
            "{json}"
        );
        assert!(
            json.contains("\"secondary_range\":{\"left\":435.728485,\"right\":430.605499}"),
            "{json}"
        );
    }
}
