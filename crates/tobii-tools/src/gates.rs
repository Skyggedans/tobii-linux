//! The acceptance gates of `compare-dll --head`: a file of per-session
//! bounds on its metrics (`tools/headpose/gates.json`), and the check of one
//! session's metrics against them.
//!
//! The file is JSON:
//!
//! ```text
//! { "about": "...", "status": "...",
//!   "sessions": [ { "name": "s1", "clock_offset_us": 9290110919, "about": "..." }, ... ],
//!   "gates": [ { "id": "G1", "name": "...", "metric": "coverage_pct", "kind": "min",
//!                "by": "compare-dll", "about": "...",
//!                "thresholds": { "s1": 99.5, "s2": 95.5, "s3": 99.5 } }, ... ] }
//! ```
//!
//! A session is known by its clock offset, the device's time less the
//! Stream Engine's (see [`crate::compare_dll::clock_offset`]), which
//! `compare-dll` works out from the two logs: there is no file name to get
//! wrong. A gate bounds one metric: `min` from below, `max` from above,
//! `abs` its magnitude from above, `range` from both sides (its thresholds
//! are `[lo, hi]`). A gate `by` `compare-dll` is checked here. One `by`
//! `python` bounds the lag or the jitter of the filters, which are measured
//! against a reference this tool does not have, in Python, from the CSV
//! `compare-dll --head` writes: it is listed, not checked.

use std::fmt;
use std::fs;

use anyhow::{Context, Result, bail, ensure};

/// A JSON value: what a gates file needs of JSON. A number keeps its text,
/// so that an integer reads back exactly.
#[derive(Debug, Clone, PartialEq)]
enum Json {
    /// `true`, `false` or `null`, which no gate reads.
    Literal,
    /// A number, as written.
    Number(String),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<Json>),
    /// An object's members, in order.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Parse a whole JSON text.
    fn parse(text: &str) -> Result<Self> {
        let mut parser = Parser {
            text: text.as_bytes(),
            at: 0,
        };
        let value = parser.value()?;
        parser.skip_space();
        ensure!(
            parser.at == parser.text.len(),
            "more after the value, at byte {}",
            parser.at
        );
        Ok(value)
    }

    /// The member `key` of an object.
    fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Object(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(text) => text.parse().ok(),
            _ => None,
        }
    }

    /// An integer, written as one.
    fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Number(text) => text.parse().ok(),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    fn as_object(&self) -> Option<&[(String, Self)]> {
        match self {
            Self::Object(members) => Some(members),
            _ => None,
        }
    }
}

/// A recursive-descent JSON parser. Numbers are taken as Rust reads them,
/// which is a little more than JSON allows (`1.`, `01`).
struct Parser<'a> {
    text: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.text.get(self.at).copied()
    }

    fn skip_space(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.at += 1;
        }
    }

    /// Step over `byte`, which must come next.
    fn eat(&mut self, byte: u8) -> Result<()> {
        ensure!(
            self.peek() == Some(byte),
            "expected '{}' at byte {}",
            char::from(byte),
            self.at
        );
        self.at += 1;
        Ok(())
    }

    fn value(&mut self) -> Result<Json> {
        self.skip_space();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(b't' | b'f' | b'n') => self.literal(),
            Some(other) => bail!("unexpected '{}' at byte {}", char::from(other), self.at),
            None => bail!("the text ends where a value should be"),
        }
    }

    fn literal(&mut self) -> Result<Json> {
        let rest = self.text.get(self.at..).unwrap_or_default();
        let Some(word) = ["true", "false", "null"]
            .into_iter()
            .find(|w| rest.starts_with(w.as_bytes()))
        else {
            bail!("unknown word at byte {}", self.at);
        };
        self.at += word.len();
        Ok(Json::Literal)
    }

    fn number(&mut self) -> Result<Json> {
        let start = self.at;
        while let Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') = self.peek() {
            self.at += 1;
        }
        let text = std::str::from_utf8(self.text.get(start..self.at).unwrap_or_default())
            .context("a number that is not text")?;
        ensure!(
            text.parse::<f64>().is_ok(),
            "bad number {text} at byte {start}"
        );
        Ok(Json::Number(text.to_string()))
    }

    fn string(&mut self) -> Result<String> {
        let start = self.at;
        self.eat(b'"')?;
        let mut bytes = Vec::new();
        loop {
            let Some(byte) = self.peek() else {
                bail!("the string at byte {start} never ends");
            };
            self.at += 1;
            match byte {
                b'"' => break,
                b'\\' => self.escape(&mut bytes)?,
                0..=0x1f => bail!("a control character in the string at byte {start}"),
                _ => bytes.push(byte),
            }
        }
        String::from_utf8(bytes).with_context(|| format!("the string at byte {start} is not UTF-8"))
    }

    /// The escape after a backslash, appended to `bytes`. `\u` takes a
    /// character of the Basic Multilingual Plane, not a surrogate.
    fn escape(&mut self, bytes: &mut Vec<u8>) -> Result<()> {
        let at = self.at;
        let Some(code) = self.peek() else {
            bail!("the text ends in an escape");
        };
        self.at += 1;
        let byte = match code {
            b'"' => b'"',
            b'\\' => b'\\',
            b'/' => b'/',
            b'b' => 0x08,
            b'f' => 0x0c,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'u' => {
                let hex = self
                    .text
                    .get(self.at..self.at + 4)
                    .and_then(|h| std::str::from_utf8(h).ok())
                    .with_context(|| format!("a short \\u escape at byte {at}"))?;
                let c = u32::from_str_radix(hex, 16)
                    .ok()
                    .and_then(char::from_u32)
                    .with_context(|| format!("a bad \\u escape at byte {at}"))?;
                self.at += 4;
                bytes.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
                return Ok(());
            }
            other => bail!("unknown escape '\\{}' at byte {at}", char::from(other)),
        };
        bytes.push(byte);
        Ok(())
    }

    fn array(&mut self) -> Result<Json> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.skip_space();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::Array(items));
                }
                _ => bail!("expected ',' or ']' at byte {}", self.at),
            }
        }
    }

    fn object(&mut self) -> Result<Json> {
        self.eat(b'{')?;
        let mut members = Vec::new();
        self.skip_space();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Json::Object(members));
        }
        loop {
            self.skip_space();
            let key = self.string()?;
            self.skip_space();
            self.eat(b':')?;
            members.push((key, self.value()?));
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Object(members));
                }
                _ => bail!("expected ',' or '}}' at byte {}", self.at),
            }
        }
    }
}

/// A gate's bound on its metric for one session.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Bound {
    /// At least this (`"kind": "min"`).
    Min(f64),
    /// At most this (`max`).
    Max(f64),
    /// At most this in magnitude (`abs`).
    Abs(f64),
    /// Within `[lo, hi]` (`range`).
    Range(f64, f64),
}

impl Bound {
    /// Whether `value` passes; a NaN never does.
    pub(crate) fn passes(self, value: f64) -> bool {
        match self {
            Self::Min(t) => value >= t,
            Self::Max(t) => value <= t,
            Self::Abs(t) => value.abs() <= t,
            Self::Range(lo, hi) => lo <= value && value <= hi,
        }
    }
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Min(t) => write!(f, ">= {t:.2}"),
            Self::Max(t) => write!(f, "<= {t:.2}"),
            Self::Abs(t) => write!(f, "|x| <= {t:.2}"),
            Self::Range(lo, hi) => write!(f, "{lo:.2}..{hi:.2}"),
        }
    }
}

/// What computes a gate's metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Checker {
    /// `compare-dll --head`, which checks the gate.
    CompareDll,
    /// The Python metrics of the filters' lag and jitter.
    Python,
}

/// One gate of a gates file.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Gate {
    /// Its id, `G1`...
    pub(crate) id: String,
    /// What it bounds, in words.
    pub(crate) name: String,
    /// The name of the metric it bounds.
    pub(crate) metric: String,
    /// What computes the metric.
    pub(crate) checker: Checker,
    /// Its bound for each session, by session name.
    bounds: Vec<(String, Bound)>,
}

impl Gate {
    /// Read one gate of a file whose sessions are `sessions`.
    fn parse(json: &Json, sessions: &[Session]) -> Result<Self> {
        let text = |key: &str| {
            json.get(key)
                .and_then(Json::as_str)
                .with_context(|| format!("no \"{key}\" string"))
        };
        let id = text("id").context("a gate")?.to_string();
        let gate = || -> Result<Self> {
            let kind = text("kind")?;
            let checker = match text("by")? {
                "compare-dll" => Checker::CompareDll,
                "python" => Checker::Python,
                other => bail!("\"by\" is compare-dll or python, not {other}"),
            };
            let thresholds = json
                .get("thresholds")
                .and_then(Json::as_object)
                .context("no \"thresholds\" object")?;
            if let Some((name, _)) = thresholds
                .iter()
                .find(|(name, _)| !sessions.iter().any(|s| &s.name == name))
            {
                bail!("a threshold for {name}, which is no session");
            }
            let bounds = sessions
                .iter()
                .map(|session| {
                    let value = thresholds
                        .iter()
                        .find(|(name, _)| *name == session.name)
                        .map(|(_, v)| v)
                        .with_context(|| format!("no threshold for {}", session.name))?;
                    let bound = bound(kind, value)
                        .with_context(|| format!("the threshold for {}", session.name))?;
                    Ok((session.name.clone(), bound))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                id: id.clone(),
                name: text("name")?.to_string(),
                metric: text("metric")?.to_string(),
                checker,
                bounds,
            })
        };
        gate().with_context(|| format!("gate {id}"))
    }

    /// Its bound for the session `name`.
    fn bound(&self, name: &str) -> Option<Bound> {
        self.bounds.iter().find(|(n, _)| n == name).map(|(_, b)| *b)
    }
}

/// The bound of a gate of `kind` whose threshold is `value`.
fn bound(kind: &str, value: &Json) -> Result<Bound> {
    let number = || value.as_f64().context("not a number");
    Ok(match kind {
        "min" => Bound::Min(number()?),
        "max" => Bound::Max(number()?),
        "abs" => Bound::Abs(number()?),
        "range" => {
            let ends = value
                .as_array()
                .and_then(|a| <[Json; 2]>::try_from(a.to_vec()).ok())
                .context("not [lo, hi]")?;
            let [lo, hi] = ends.map(|e| e.as_f64());
            let (Some(lo), Some(hi)) = (lo, hi) else {
                bail!("not [lo, hi]");
            };
            ensure!(lo <= hi, "[{lo}, {hi}] is empty");
            Bound::Range(lo, hi)
        }
        other => bail!("\"kind\" is min, max, abs or range, not {other}"),
    })
}

/// A session of a gates file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Session {
    /// Its name, which the thresholds go by.
    pub(crate) name: String,
    /// Its clock offset, µs, which tells it.
    pub(crate) clock_offset_us: i64,
}

/// A gates file.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Gates {
    /// What the file says of its own standing: whether its bounds are
    /// provisional, say.
    pub(crate) status: String,
    /// The sessions it has bounds for.
    pub(crate) sessions: Vec<Session>,
    /// Its gates, in order.
    pub(crate) gates: Vec<Gate>,
}

/// One gate checked on one session.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Checked<'a> {
    /// The gate.
    pub(crate) gate: &'a Gate,
    /// Its bound for the session.
    pub(crate) bound: Bound,
    /// The metric's value, for a gate checked here.
    pub(crate) value: Option<f64>,
}

impl Checked<'_> {
    /// Whether the gate is checked here and fails.
    pub(crate) fn fails(&self) -> bool {
        self.value.is_some_and(|v| !self.bound.passes(v))
    }
}

/// The verdict of the gates `checked`: the run passes when none fails.
///
/// # Errors
/// Names every gate that fails, in order; `compare-dll` then exits with a
/// status other than 0, as design §10 asks.
pub(crate) fn verdict(checked: &[Checked<'_>]) -> Result<()> {
    let failed: Vec<&str> = checked
        .iter()
        .filter(|c| c.fails())
        .map(|c| c.gate.id.as_str())
        .collect();
    ensure!(failed.is_empty(), "gates failed: {}", failed.join(", "));
    Ok(())
}

impl Gates {
    /// Read the gates file at `path`.
    ///
    /// # Errors
    /// Fails when the file cannot be read, or is not a gates file (see the
    /// [module docs](self)).
    pub(crate) fn load(path: &str) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| format!("failed to read {path}"))?;
        Self::parse(&text).with_context(|| format!("bad gates file {path}"))
    }

    /// Read a gates file's text.
    ///
    /// # Errors
    /// Fails when `text` is not a gates file (see the [module docs](self)).
    pub(crate) fn parse(text: &str) -> Result<Self> {
        let json = Json::parse(text)?;
        let sessions = json
            .get("sessions")
            .and_then(Json::as_array)
            .context("no \"sessions\" array")?
            .iter()
            .map(|s| {
                let name = s
                    .get("name")
                    .and_then(Json::as_str)
                    .context("a session without a \"name\" string")?;
                let clock_offset_us = s
                    .get("clock_offset_us")
                    .and_then(Json::as_i64)
                    .with_context(|| format!("session {name}: no integer \"clock_offset_us\""))?;
                Ok(Session {
                    name: name.to_string(),
                    clock_offset_us,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(!sessions.is_empty(), "no sessions");
        for (i, s) in sessions.iter().enumerate() {
            ensure!(
                !sessions[..i]
                    .iter()
                    .any(|t| t.name == s.name || t.clock_offset_us == s.clock_offset_us),
                "session {} repeats a name or a clock offset",
                s.name
            );
        }
        let gates = json
            .get("gates")
            .and_then(Json::as_array)
            .context("no \"gates\" array")?
            .iter()
            .map(|g| Gate::parse(g, &sessions))
            .collect::<Result<Vec<_>>>()?;
        for (i, g) in gates.iter().enumerate() {
            ensure!(
                !gates[..i].iter().any(|h| h.id == g.id),
                "gate {} appears twice",
                g.id
            );
        }
        Ok(Self {
            status: json
                .get("status")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_string(),
            sessions,
            gates,
        })
    }

    /// The session of clock offset `k_us`.
    ///
    /// # Errors
    /// Fails when the file has none.
    pub(crate) fn session(&self, k_us: i64) -> Result<&Session> {
        self.sessions
            .iter()
            .find(|s| s.clock_offset_us == k_us)
            .with_context(|| {
                let known: Vec<String> = self
                    .sessions
                    .iter()
                    .map(|s| format!("{} ({} us)", s.name, s.clock_offset_us))
                    .collect();
                format!(
                    "the gates file has no session of clock offset {k_us} us, only {}",
                    known.join(", ")
                )
            })
    }

    /// Check every gate on `session`, reading the metric of each gate
    /// checked here with `metric`.
    ///
    /// # Errors
    /// Fails for a gate checked here whose metric `metric` does not give,
    /// and for a session not of this file.
    pub(crate) fn check<'a>(
        &'a self,
        session: &Session,
        metric: impl Fn(&str) -> Option<f64>,
    ) -> Result<Vec<Checked<'a>>> {
        self.gates
            .iter()
            .map(|gate| {
                let bound = gate
                    .bound(&session.name)
                    .with_context(|| format!("no session {} in the gates file", session.name))?;
                let value = match gate.checker {
                    Checker::Python => None,
                    Checker::CompareDll => Some(metric(&gate.metric).with_context(|| {
                        format!(
                            "gate {}: compare-dll has no metric {}",
                            gate.id, gate.metric
                        )
                    })?),
                };
                Ok(Checked { gate, bound, value })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_values_parse() {
        let json = Json::parse(
            r#" { "a": [1, -2.5e3, "x\"\\\/é\n"], "b": {}, "c": [], "d": true, "e": null, "f": "√" } "#,
        )
        .expect("valid JSON");
        let a = json.get("a").and_then(Json::as_array).expect("an array");
        assert_eq!(a[0].as_i64(), Some(1));
        assert_eq!(a[1].as_f64(), Some(-2500.0));
        assert_eq!(a[1].as_i64(), None);
        assert_eq!(a[2].as_str(), Some("x\"\\/é\n"));
        assert_eq!(json.get("b").and_then(Json::as_object), Some(&[][..]));
        assert_eq!(json.get("c").and_then(Json::as_array), Some(&[][..]));
        assert_eq!(json.get("d"), Some(&Json::Literal));
        assert_eq!(json.get("f").and_then(Json::as_str), Some("√"));
        assert_eq!(json.get("g"), None);
        // Integers read back exactly, beyond f64's 2^53.
        assert_eq!(
            Json::parse("9007199254740993")
                .ok()
                .and_then(|j| j.as_i64()),
            Some(9_007_199_254_740_993)
        );
    }

    #[test]
    fn broken_json_is_refused() {
        for text in [
            "",
            "{",
            "[1,]",
            "[1 2]",
            r#"{"a" 1}"#,
            r#"{"a": 1,}"#,
            r#""open"#,
            r#""\x""#,
            r#""\ud800""#,
            "\"a\u{1}\"",
            "1 2",
            "nul",
            "-",
            "1e",
        ] {
            assert!(Json::parse(text).is_err(), "{text:?} parsed");
        }
    }

    /// A gates file of two sessions and one gate of each kind, one of them
    /// Python's.
    const FILE: &str = r#"{
        "about": "test", "status": "provisional",
        "sessions": [
            {"name": "s1", "clock_offset_us": 9290110919, "about": "first"},
            {"name": "s2", "clock_offset_us": 10701480021}
        ],
        "gates": [
            {"id": "G1", "name": "coverage", "metric": "coverage_pct", "kind": "min",
             "by": "compare-dll", "about": "...", "thresholds": {"s1": 99.5, "s2": 95.5}},
            {"id": "G4", "name": "pitch", "metric": "rotation_median_abs_x_deg", "kind": "max",
             "by": "compare-dll", "thresholds": {"s1": 2.3, "s2": 2.6}},
            {"id": "G7", "name": "bias", "metric": "rotation_median_signed_x_deg", "kind": "abs",
             "by": "compare-dll", "thresholds": {"s1": 2.1, "s2": 1.6}},
            {"id": "G17", "name": "jitter", "metric": "rotation_rest_jitter_ratio", "kind": "range",
             "by": "python", "thresholds": {"s1": [0.6, 1.5], "s2": [0.6, 1.6]}}
        ]
    }"#;

    #[test]
    fn a_gates_file_reads() {
        let gates = Gates::parse(FILE).expect("a gates file");
        assert_eq!(gates.status, "provisional");
        assert_eq!(
            gates.sessions,
            [
                Session {
                    name: "s1".into(),
                    clock_offset_us: 9_290_110_919,
                },
                Session {
                    name: "s2".into(),
                    clock_offset_us: 10_701_480_021,
                },
            ]
        );
        assert_eq!(gates.gates.len(), 4);
        assert_eq!(gates.gates[3].checker, Checker::Python);
        assert_eq!(gates.gates[3].bound("s2"), Some(Bound::Range(0.6, 1.6)));
        assert_eq!(gates.gates[2].bound("s1"), Some(Bound::Abs(2.1)));
        assert_eq!(
            gates.session(10_701_480_021).map(|s| s.name.as_str()).ok(),
            Some("s2")
        );
        assert!(gates.session(12_670_044_474).is_err());
    }

    #[test]
    fn bounds_pass_and_fail() {
        assert!(Bound::Min(95.5).passes(95.5) && !Bound::Min(95.5).passes(95.49));
        assert!(Bound::Max(2.3).passes(2.3) && !Bound::Max(2.3).passes(2.31));
        assert!(Bound::Abs(2.1).passes(-2.1) && !Bound::Abs(2.1).passes(-2.11));
        assert!(Bound::Range(0.6, 1.5).passes(0.6) && Bound::Range(0.6, 1.5).passes(1.5));
        assert!(!Bound::Range(0.6, 1.5).passes(0.59) && !Bound::Range(0.6, 1.5).passes(1.51));
        for bound in [
            Bound::Min(0.0),
            Bound::Max(0.0),
            Bound::Abs(0.0),
            Bound::Range(0.0, 1.0),
        ] {
            assert!(!bound.passes(f64::NAN), "{bound} passed a NaN");
        }
        assert_eq!(Bound::Abs(2.1).to_string(), "|x| <= 2.10");
        assert_eq!(Bound::Range(0.6, 1.5).to_string(), "0.60..1.50");
    }

    /// Session 2's metrics checked: coverage and bias pass, pitch fails;
    /// the Python gate is listed with its bound and no value, even should
    /// compare-dll have a metric of that name.
    #[test]
    fn a_session_is_checked_gate_by_gate() {
        let gates = Gates::parse(FILE).expect("a gates file");
        let session = gates.session(10_701_480_021).expect("session s2").clone();
        let metric = |name: &str| match name {
            "coverage_pct" => Some(96.79),
            "rotation_median_abs_x_deg" => Some(2.61),
            "rotation_median_signed_x_deg" => Some(-1.59),
            "rotation_rest_jitter_ratio" => Some(9.0),
            _ => None,
        };
        let checked = gates.check(&session, metric).expect("every metric known");
        let verdicts: Vec<(&str, Option<f64>, bool)> = checked
            .iter()
            .map(|c| (c.gate.id.as_str(), c.value, c.fails()))
            .collect();
        assert_eq!(
            verdicts,
            [
                ("G1", Some(96.79), false),
                ("G4", Some(2.61), true),
                ("G7", Some(-1.59), false),
                ("G17", None, false),
            ]
        );
        assert_eq!(checked[1].bound, Bound::Max(2.6));
        let failed = verdict(&checked).expect_err("G4 fails");
        assert_eq!(failed.to_string(), "gates failed: G4");

        // A metric compare-dll does not have is an error, not a pass.
        let partial = |name: &str| (name == "coverage_pct").then_some(99.0);
        assert!(gates.check(&session, partial).is_err());
    }

    /// The run passes when every gate checked here does, whatever the
    /// Python gates' metrics; it fails naming each gate that fails.
    #[test]
    fn the_verdict_fails_the_run_for_each_failing_gate() {
        let gates = Gates::parse(FILE).expect("a gates file");
        let session = gates.session(10_701_480_021).expect("session s2").clone();
        let at = |coverage: f64, pitch: f64| {
            move |name: &str| match name {
                "coverage_pct" => Some(coverage),
                "rotation_median_abs_x_deg" => Some(pitch),
                "rotation_median_signed_x_deg" => Some(0.0),
                _ => None,
            }
        };
        let checked = gates.check(&session, at(96.79, 2.6)).expect("metrics");
        assert!(verdict(&checked).is_ok());
        let checked = gates.check(&session, at(95.0, 2.61)).expect("metrics");
        let failed = verdict(&checked).expect_err("G1 and G4 fail");
        assert_eq!(failed.to_string(), "gates failed: G1, G4");
        assert!(verdict(&[]).is_ok());
    }

    #[test]
    fn malformed_gates_files_are_refused() {
        let swap = |from: &str, to: &str| {
            assert!(FILE.contains(from), "{from}");
            FILE.replacen(from, to, 1)
        };
        for text in [
            // A session without a threshold, a threshold for no session.
            swap(r#", "s2": 2.6}"#, "}"),
            swap(r#""s2": 95.5}"#, r#""s2": 95.5, "s9": 90.0}"#),
            // A range that is a number, a number that is a range, an
            // empty range.
            swap("[0.6, 1.5]", "0.6"),
            swap("99.5", "[99.5, 100]"),
            swap("[0.6, 1.6]", "[1.6, 0.6]"),
            // An unknown kind or checker, a clock offset that is no
            // integer, a repeated session or gate.
            swap(r#""kind": "max""#, r#""kind": "most""#),
            swap(r#""by": "python""#, r#""by": "matlab""#),
            swap("9290110919", "9290110919.5"),
            swap(r#""name": "s2""#, r#""name": "s1""#),
            swap(r#""id": "G7""#, r#""id": "G4""#),
        ] {
            assert!(Gates::parse(&text).is_err(), "{text}");
        }
    }
}
