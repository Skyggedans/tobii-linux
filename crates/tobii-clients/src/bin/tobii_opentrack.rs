//! Thin client: subscribe to the Stream Engine's head pose from `tobiid` and
//! send it to `OpenTrack`'s *UDP over network* input, the 6-double
//! `x, y, z, yaw, pitch, roll` packet. Auto-spawns the daemon if it isn't
//! running.
//!
//! Each axis goes to `OpenTrack` as its own `tracker-tobii` plugin hands over
//! the same pose through `libtobii.so` ([`opentrack_pose`]), so the two
//! inputs move alike and one `OpenTrack` profile fits both. The pose is
//! absolute: `OpenTrack` centres it, at the first pose (*Center at startup*)
//! and on its own *Center* shortcut.

use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tobii_ipc::{
    HeadPose, STREAM_HEAD_POSE, ServerMsg, decode_server, encode_subscribe, read_frame, write_frame,
};
use tracing::warn;

/// Millimetres in a centimetre: tobiid sends the position in mm, `OpenTrack`
/// takes it in cm.
const MM_PER_CM: f64 = 10.0;

/// How long the head pose may stay silent after the subscription before the
/// bridge warns, once, that tobiid may not send it; as long as `libtobii.so`
/// waits. A current tobiid sends a pose for every IR image, valid or not, but
/// none before the tracker streams, which a cold start takes ~12 s (at worst
/// about 15) to reach.
const HEAD_POSE_SILENCE: Duration = Duration::from_secs(20);

/// What `--help` prints.
const USAGE: &str = "\
usage: tobii-opentrack [--host 127.0.0.1] [--port 4242]

Sends the Stream Engine's head pose from tobiid to OpenTrack's \"UDP over
network\" input: x, y, z in cm and yaw, pitch, roll in degrees, each axis as
OpenTrack's own tracker-tobii plugin hands it over. A pose tobiid marks
invalid (no face, or one at the edge of the camera's view) is not sent, and
OpenTrack holds the last one.

The pose is absolute: OpenTrack centres it, at the first pose (\"Center at
startup\") and on its own Center shortcut.";

/// The error `--recenter` gets: an old hotkey script that passes it is told
/// what replaced it, rather than starting a second bridge.
const RECENTER_GONE: &str = "--recenter is gone: the head pose is the Stream Engine's, \
                             absolute, with no rest pose in tobiid to reset; centre it with \
                             OpenTrack's own Center shortcut";

fn main() {
    tobii_log::init();
    if let Err(e) = run() {
        eprintln!("tobii-opentrack: {e:#}");
        std::process::exit(1);
    }
}

/// What the command line asks for.
#[derive(Debug, PartialEq, Eq)]
enum Cli {
    /// Print [`USAGE`] and exit.
    Help,
    /// Send the head pose to `OpenTrack` at `host:port`.
    Bridge {
        /// Where `OpenTrack` listens.
        host: String,
        /// Its UDP port.
        port: u16,
    },
}

/// Read the command line, the program name left out.
///
/// # Errors
///
/// An option without its value, a port that does not parse, `--recenter`
/// ([`RECENTER_GONE`]) or an argument it does not know.
fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Cli> {
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 4242;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--host" => host = args.next().context("--host needs a value")?,
            "--port" => {
                let v = args.next().context("--port needs a value")?;
                port = v
                    .parse()
                    .with_context(|| format!("--port: invalid value {v:?}"))?;
            }
            "--recenter" => bail!(RECENTER_GONE),
            "-h" | "--help" => return Ok(Cli::Help),
            other => bail!("unknown arg: {other}"),
        }
    }
    Ok(Cli::Bridge { host, port })
}

/// `OpenTrack`'s pose for a head pose from tobiid, in its UDP packet's
/// order: TX, TY, TZ in cm, then yaw, pitch, roll in degrees. `None` unless
/// the position and all three angles are valid.
///
/// Each axis is what `OpenTrack`'s `tracker-tobii` plugin makes of the
/// Stream Engine's pose (`tobii_tracker::data`): of the position in the
/// display frame, TX is -x, TY y and TZ z, in mm over 10; of the rotation,
/// yaw is minus the angle about y, pitch the angle about x and roll the
/// angle about z, in degrees.
///
/// An invalid pose gives nothing to send: `OpenTrack`'s UDP input then holds
/// the last packet, as the plugin holds each axis's last valid value. The
/// input holds a packet whole, so a pose with any of its four validities
/// clear is not sent; the Stream Engine and tobiid set the four together.
fn opentrack_pose(pose: &HeadPose) -> Option<[f64; 6]> {
    if !(pose.position_valid && pose.rotation_valid.iter().all(|&v| v)) {
        return None;
    }
    let [x, y, z] = pose.position_mm.map(|mm| f64::from(mm) / MM_PER_CM);
    let [about_x, about_y, about_z] = pose.rotation_rad.map(|rad| f64::from(rad).to_degrees());
    Some([-x, y, z, -about_y, about_x, about_z])
}

/// The UDP packet for `OpenTrack`'s input: the six values of
/// [`opentrack_pose`] in order, each a little-endian `f64`, the `double[6]`
/// it reads on a little-endian host.
fn udp_packet(pose: &[f64; 6]) -> [u8; 48] {
    let mut packet = [0u8; 48];
    for (chunk, v) in packet.as_chunks_mut::<8>().0.iter_mut().zip(pose) {
        *chunk = v.to_le_bytes();
    }
    packet
}

/// How the bridge reaches `OpenTrack`'s UDP input ([`udp_route`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UdpRoute {
    /// The address the bridge's socket is bound to: the unspecified one of
    /// the target's family, any port.
    local: SocketAddr,
    /// Where `OpenTrack` listens.
    target: SocketAddr,
}

/// The route to `OpenTrack` among the addresses its host resolved to, or
/// `None` if there are none. The target is the first IPv4 address if there
/// is one, else the first address: a receiver bound to IPv4, or to both
/// families as `OpenTrack`'s UDP input is, takes IPv4. The socket is bound
/// in the target's family, as one of the other family cannot send there:
/// `localhost` resolves to `::1` first on many hosts, and a socket bound to
/// `0.0.0.0` gets `EAFNOSUPPORT` sending to it.
fn udp_route(resolved: &[SocketAddr]) -> Option<UdpRoute> {
    let target = *resolved
        .iter()
        .find(|addr| addr.is_ipv4())
        .or_else(|| resolved.first())?;
    let local = if target.is_ipv4() {
        SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))
    } else {
        SocketAddr::from((Ipv6Addr::UNSPECIFIED, 0))
    };
    Some(UdpRoute { local, target })
}

/// Wait `window`, then, if no head pose has come by then (`seen` still
/// clear), warn that tobiid may not send one. The thread's result is
/// whether it warned.
fn watch_for_silence(seen: Arc<AtomicBool>, window: Duration) -> JoinHandle<bool> {
    thread::spawn(move || {
        thread::sleep(window);
        let silent = !seen.load(Ordering::Relaxed);
        if silent {
            warn!(
                "tobiid has sent no head pose in the {} s since it was subscribed: a tobiid \
                 older than this tobii-opentrack sends none (restart it after installing), \
                 nor does one whose tracker sends no IR images (unplugged, paused, or with \
                 TOBII_NO_IMAGE set)",
                window.as_secs()
            );
        }
        silent
    })
}

fn run() -> Result<()> {
    let (host, port) = match parse_args(std::env::args().skip(1))? {
        Cli::Help => {
            println!("{USAGE}");
            return Ok(());
        }
        Cli::Bridge { host, port } => (host, port),
    };

    // Resolve once up front: `send_to` with a host string would re-resolve it
    // for every frame.
    let resolved: Vec<SocketAddr> = (host.as_str(), port)
        .to_socket_addrs()
        .with_context(|| format!("resolving {host}:{port}"))?
        .collect();
    let UdpRoute { local, target } =
        udp_route(&resolved).with_context(|| format!("{host}:{port} resolved to no address"))?;
    let socket =
        UdpSocket::bind(local).with_context(|| format!("binding a udp socket to {local}"))?;

    let mut stream = tobii_ipc::connect_or_spawn().context("connecting to tobiid")?;
    write_frame(&mut stream, &encode_subscribe(STREAM_HEAD_POSE))
        .context("subscribing to the head pose stream")?;
    let seen = Arc::new(AtomicBool::new(false));
    // Left to run on its own: only an error ends the loop below, and with it
    // the process.
    drop(watch_for_silence(Arc::clone(&seen), HEAD_POSE_SILENCE));

    println!("tobii-opentrack: head pose -> OpenTrack {host}:{port}");
    let mut sent = 0u64;
    loop {
        let Some(body) = read_frame(&mut stream).context("reading from tobiid")? else {
            bail!("daemon closed the connection");
        };
        match decode_server(&body) {
            Some(ServerMsg::Subscribed { ok: false }) => {
                bail!("daemon is busy with the other mode (gaze)")
            }
            Some(ServerMsg::HeadPose(head)) => {
                seen.store(true, Ordering::Relaxed);
                let Some(pose) = opentrack_pose(&head) else {
                    continue;
                };
                socket
                    .send_to(&udp_packet(&pose), target)
                    .context("sending to OpenTrack")?;
                sent += 1;
                if sent.is_multiple_of(8) {
                    eprint!(
                        "\ryaw/pit/roll={:+5.1}/{:+5.1}/{:+5.1}  xyz={:+6.1}/{:+6.1}/{:+6.1}   ",
                        pose[3], pose[4], pose[5], pose[0], pose[1], pose[2]
                    );
                }
            }
            // The SUBSCRIBED ack, and frames of streams not subscribed here;
            // unknown tags decode to `None`. `ServerMsg` belongs to the
            // library crate (this binary is a separate crate), so a wildcard
            // keeps this client building if the enum grows or becomes
            // `#[non_exhaustive]`.
            _ => {}
        }
    }
}

#[cfg(test)]
// reason: err-no-unwrap-prod exempts test code; a failed unwrap here is the test failing.
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A pose with every validity set and the given values.
    fn valid(position_mm: [f32; 3], rotation_rad: [f32; 3]) -> HeadPose {
        HeadPose {
            ts_us: 1,
            position_valid: true,
            position_mm,
            rotation_valid: [true; 3],
            rotation_rad,
        }
    }

    fn assert_close(got: [f64; 6], want: [f64; 6]) {
        assert!(
            got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-6),
            "got {got:?}, want {want:?}"
        );
    }

    /// Each value of the pose moves the one `OpenTrack` axis the plugin
    /// gives it, with the plugin's sign: +x is -TX, +y TY, +z TZ, +rotation
    /// about x pitch, about y -yaw and about z roll.
    #[test]
    fn each_value_moves_the_plugins_axis_with_its_sign() {
        // 10 mm of position is 1 cm, 0.25 rad of rotation 14.32°.
        let deg = 14.323_944_878;
        // (the value set: x, y, z, then the angle about x, y, z; the axis
        // that moves: TX, TY, TZ, yaw, pitch, roll; how far it moves)
        let cases = [
            (0, 0, -1.0),
            (1, 1, 1.0),
            (2, 2, 1.0),
            (3, 4, deg),
            (4, 3, -deg),
            (5, 5, deg),
        ];
        for (value, axis, moved) in cases {
            let mut values = [0.0; 6];
            values[value] = if value < 3 { 10.0 } else { 0.25 };
            let [x, y, z, about_x, about_y, about_z] = values;
            let mut want = [0.0; 6];
            want[axis] = moved;
            let pose = valid([x, y, z], [about_x, about_y, about_z]);
            assert_close(opentrack_pose(&pose).unwrap(), want);
        }
    }

    /// Millimetres go out as centimetres and radians as degrees, every axis
    /// at once, none taken for another.
    #[test]
    fn a_pose_goes_out_in_centimetres_and_degrees() {
        let position_mm = [-31.5, 84.0, 571.5];
        let rotation_rad = [0.184_108_84, -0.037_300_18, -0.049_585_36];
        let want = [
            3.15,           // TX: -x
            8.4,            // TY: y
            57.15,          // TZ: z
            2.137_142_926,  // yaw: minus the angle about y
            10.548_659_414, // pitch: the angle about x
            -2.841_031_913, // roll: the angle about z
        ];
        let pose = valid(position_mm, rotation_rad);
        assert_close(opentrack_pose(&pose).unwrap(), want);
    }

    /// An invalid pose, or one with any validity clear, gives nothing to
    /// send, whatever its values read; a valid pose at the origin is sent.
    #[test]
    fn a_pose_is_sent_only_with_every_validity_set() {
        let pose = valid([12.0, -3.0, 600.0], [0.1, -0.2, 0.05]);
        let invalid = HeadPose {
            position_valid: false,
            rotation_valid: [false; 3],
            ..pose
        };
        assert_eq!(opentrack_pose(&invalid), None);
        assert_eq!(
            opentrack_pose(&HeadPose {
                position_valid: false,
                ..pose
            }),
            None
        );
        for axis in 0..3 {
            let mut rotation_valid = [true; 3];
            rotation_valid[axis] = false;
            let pose = HeadPose {
                rotation_valid,
                ..pose
            };
            assert_eq!(opentrack_pose(&pose), None, "rotation {axis} invalid");
        }
        assert!(opentrack_pose(&pose).is_some());
        assert_eq!(opentrack_pose(&valid([0.0; 3], [0.0; 3])), Some([0.0; 6]));
    }

    /// The packet is the six values in order, little-endian `f64`s.
    #[test]
    fn the_packet_is_six_little_endian_doubles_in_order() {
        let pose = [1.0, -2.5, 60.0, -10.0, 5.5, 0.125];
        let packet = udp_packet(&pose);
        assert_eq!(&packet[..8], &[0, 0, 0, 0, 0, 0, 0xf0, 0x3f]);
        let read: Vec<f64> = packet
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect();
        assert_eq!(read, pose);
    }

    /// The packets go to the first IPv4 address the host resolved to, else
    /// to its first address, from the unspecified address of the target's
    /// family; a host that resolved to nothing has no route.
    #[test]
    fn packets_go_to_ipv4_first_from_the_targets_family() {
        let addr = |s: &str| -> SocketAddr { s.parse().unwrap() };
        let (any4, any6) = (addr("0.0.0.0:0"), addr("[::]:0"));
        let (lo4, lo6) = (addr("127.0.0.1:4242"), addr("[::1]:4242"));
        let (lan4, lan6) = (addr("10.0.0.2:4242"), addr("[fd00::2]:4242"));
        let route = |local, target| Some(UdpRoute { local, target });
        // (what the host resolved to, the route)
        let cases = [
            // `localhost` on many hosts, this one included.
            (vec![lo6, lo4], route(any4, lo4)),
            (vec![lo4, lo6], route(any4, lo4)),
            (vec![lan6, lo6, lan4, lo4], route(any4, lan4)),
            (vec![lo4], route(any4, lo4)),
            (vec![lo6], route(any6, lo6)),
            (vec![lan6, lo6], route(any6, lan6)),
            (vec![], None),
        ];
        for (resolved, want) in cases {
            assert_eq!(udp_route(&resolved), want, "{resolved:?}");
        }
    }

    /// A socket bound as its route says reaches a receiver on the IPv4
    /// loopback and on the IPv6 one, `localhost`'s first address on many
    /// hosts, which a socket bound to `0.0.0.0` cannot send to.
    #[test]
    fn a_socket_bound_for_its_route_reaches_either_loopback() {
        let packet = udp_packet(&[1.0, -2.5, 60.0, -10.0, 5.5, 0.125]);
        for receiver in ["127.0.0.1:0", "[::1]:0"] {
            let receiver = match UdpSocket::bind(receiver) {
                Ok(socket) => socket,
                // A host without an IPv6 loopback has no receiver to reach.
                Err(e) if receiver.starts_with('[') => {
                    eprintln!("no IPv6 loopback to test with: {e}");
                    continue;
                }
                Err(e) => panic!("binding a receiver on {receiver}: {e}"),
            };
            receiver
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let route = udp_route(&[receiver.local_addr().unwrap()]).unwrap();

            let sender = UdpSocket::bind(route.local).unwrap();
            sender.send_to(&packet, route.target).unwrap();

            let mut got = [0u8; 64];
            let (len, from) = receiver.recv_from(&mut got).unwrap();
            assert_eq!(&got[..len], &packet, "{route:?}");
            assert_eq!(from.port(), sender.local_addr().unwrap().port());
        }
    }

    /// The command line: the defaults, `--host` and `--port`, `--help`, and
    /// `--recenter` refused with what replaced it.
    #[test]
    fn the_command_line_refuses_recenter_and_names_its_replacement() {
        let parse = |args: &[&str]| parse_args(args.iter().map(ToString::to_string));
        let bridge = |host: &str, port| Cli::Bridge {
            host: host.to_string(),
            port,
        };
        assert_eq!(parse(&[]).unwrap(), bridge("127.0.0.1", 4242));
        assert_eq!(
            parse(&["--host", "10.0.0.2", "--port", "5555"]).unwrap(),
            bridge("10.0.0.2", 5555)
        );
        assert_eq!(parse(&["--port", "4243", "-h"]).unwrap(), Cli::Help);
        assert_eq!(parse(&["--help"]).unwrap(), Cli::Help);

        for args in [&["--recenter"][..], &["--port", "4243", "--recenter"]] {
            let error = parse(args).unwrap_err().to_string();
            assert_eq!(error, RECENTER_GONE, "{args:?}");
        }
        assert!(RECENTER_GONE.contains("Center shortcut") && USAGE.contains("Center shortcut"));
        assert!(parse(&["--port", "x"]).is_err());
        assert!(parse(&["--host"]).is_err());
        assert!(parse(&["--frobnicate"]).is_err());
    }

    /// The silence warning comes only when no head pose came within the
    /// window.
    #[test]
    fn silence_is_reported_only_when_no_pose_came() {
        let seen = Arc::new(AtomicBool::new(false));
        let silent = watch_for_silence(Arc::clone(&seen), Duration::ZERO).join();
        assert!(silent.unwrap(), "no pose came");
        seen.store(true, Ordering::Relaxed);
        let silent = watch_for_silence(seen, Duration::ZERO).join();
        assert!(!silent.unwrap(), "a pose came");
    }
}
