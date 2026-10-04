//! Thin client: subscribe to head pose from `tobiid` and forward it to `OpenTrack`
//! over UDP (the 6-double `x,y,z,yaw,pitch,roll` packet). Auto-spawns the daemon
//! if it isn't running.

use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};

use anyhow::{Context, Result, bail};
use tobii_ipc::{
    self, STREAM_HEAD, ServerMsg, decode_server, encode_subscribe, read_frame, write_frame,
};

fn main() {
    tobii_log::init();
    if let Err(e) = run() {
        eprintln!("tobii-opentrack: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 4242;
    let mut recenter = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--host" => host = args.next().context("--host needs a value")?,
            "--port" => {
                let v = args.next().context("--port needs a value")?;
                port = v
                    .parse()
                    .with_context(|| format!("--port: invalid value {v:?}"))?;
            }
            "--recenter" => recenter = true,
            "-h" | "--help" => {
                println!("usage: tobii-opentrack [--host 127.0.0.1] [--port 4242] [--recenter]");
                return Ok(());
            }
            other => bail!("unknown arg: {other}"),
        }
    }

    // One-shot: ask the running daemon to recalibrate the rest pose and exit.
    // Bind this to a hotkey to re-center without restarting anything.
    if recenter {
        let mut stream = tobii_ipc::connect().context("connecting to tobiid")?;
        write_frame(&mut stream, &tobii_ipc::encode_recenter()).context("sending recenter")?;
        println!("tobii-opentrack: recenter sent");
        return Ok(());
    }

    let socket = UdpSocket::bind("0.0.0.0:0").context("binding a udp socket")?;
    // Resolve once up front: `send_to` with a host string would re-resolve it
    // for every frame.
    let target: SocketAddr = (host.as_str(), port)
        .to_socket_addrs()
        .with_context(|| format!("resolving {host}:{port}"))?
        .next()
        .with_context(|| format!("{host}:{port} resolved to no address"))?;

    let mut stream = tobii_ipc::connect_or_spawn().context("connecting to tobiid")?;
    write_frame(&mut stream, &encode_subscribe(STREAM_HEAD))
        .context("subscribing to the head stream")?;

    println!("tobii-opentrack: head pose -> OpenTrack {host}:{port}");
    let mut frames = 0u64;
    let mut packet = [0u8; 48];
    loop {
        let Some(body) = read_frame(&mut stream).context("reading from tobiid")? else {
            bail!("daemon closed the connection");
        };
        match decode_server(&body) {
            Some(ServerMsg::Subscribed { ok: false }) => {
                bail!("daemon is busy with the other mode (gaze)")
            }
            Some(ServerMsg::Subscribed { ok: true }) => {}
            Some(ServerMsg::Head {
                pos_mm, rot_rad, ..
            }) => {
                // OpenTrack order: x, y, z, yaw, pitch, roll. Translation in cm
                // (the wire carries Tobii-convention mm; OpenTrack here expects
                // cm, matching the standalone `track` path), angles in degrees.
                let pose = [
                    f64::from(pos_mm[0]) / 10.0,
                    f64::from(pos_mm[1]) / 10.0,
                    f64::from(pos_mm[2]) / 10.0,
                    f64::from(rot_rad[1]).to_degrees(), // yaw
                    f64::from(rot_rad[0]).to_degrees(), // pitch
                    f64::from(rot_rad[2]).to_degrees(), // roll
                ];
                for (chunk, v) in packet.as_chunks_mut::<8>().0.iter_mut().zip(&pose) {
                    *chunk = v.to_le_bytes();
                }
                socket
                    .send_to(&packet, target)
                    .context("sending to OpenTrack")?;
                frames += 1;
                if frames.is_multiple_of(8) {
                    eprint!(
                        "\ryaw/pit/roll={:+5.1}/{:+5.1}/{:+5.1}  xyz={:+6.1}/{:+6.1}/{:+6.1}   ",
                        pose[3], pose[4], pose[5], pose[0], pose[1], pose[2]
                    );
                }
            }
            // Gaze/presence frames aren't subscribed here and unknown tags decode
            // to `None`. `ServerMsg` belongs to the library crate (this binary is
            // a separate crate), so a wildcard keeps this client building if the
            // enum grows or becomes `#[non_exhaustive]`.
            _ => {}
        }
    }
}
