//! Thin client: subscribe to head pose from `tobiid` and forward it to OpenTrack
//! over UDP (the 6-double `x,y,z,yaw,pitch,roll` packet). Auto-spawns the daemon
//! if it isn't running.

use std::net::UdpSocket;

use tobii::ipc::{self, decode_server, encode_subscribe, read_frame, write_frame, ServerMsg, STREAM_HEAD};

fn main() {
    if let Err(e) = run() {
        eprintln!("tobii-opentrack: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 4242;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--host" => host = args.next().ok_or("--host needs a value")?,
            "--port" => port = args.next().ok_or("--port needs a value")?.parse()?,
            "-h" | "--help" => {
                println!("usage: tobii-opentrack [--host 127.0.0.1] [--port 4242]");
                return Ok(());
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    let socket = UdpSocket::bind("0.0.0.0:0")?;
    let target = format!("{host}:{port}");

    let mut stream = ipc::connect_or_spawn()?;
    write_frame(&mut stream, &encode_subscribe(STREAM_HEAD))?;

    println!("tobii-opentrack: head pose -> OpenTrack {target}");
    let mut frames = 0u64;
    loop {
        let Some(body) = read_frame(&mut stream)? else {
            return Err("daemon closed the connection".into());
        };
        match decode_server(&body) {
            Some(ServerMsg::Subscribed { ok: false }) => {
                return Err("daemon is busy with the other mode (gaze)".into());
            }
            Some(ServerMsg::Subscribed { ok: true }) => {}
            Some(ServerMsg::Head { pos_mm, rot_rad, .. }) => {
                // OpenTrack order: x, y, z, yaw, pitch, roll. Translation in cm
                // (the wire carries Tobii-convention mm; OpenTrack here expects
                // cm, matching the standalone `track` path), angles in degrees.
                let pose = [
                    pos_mm[0] as f64 / 10.0,
                    pos_mm[1] as f64 / 10.0,
                    pos_mm[2] as f64 / 10.0,
                    (rot_rad[1] as f64).to_degrees(), // yaw
                    (rot_rad[0] as f64).to_degrees(), // pitch
                    (rot_rad[2] as f64).to_degrees(), // roll
                ];
                let mut packet = [0u8; 48];
                for (i, v) in pose.iter().enumerate() {
                    packet[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
                }
                socket.send_to(&packet, &target)?;
                frames += 1;
                if frames % 8 == 0 {
                    eprint!(
                        "\ryaw/pit/roll={:+5.1}/{:+5.1}/{:+5.1}  xyz={:+6.1}/{:+6.1}/{:+6.1}   ",
                        pose[3], pose[4], pose[5], pose[0], pose[1], pose[2]
                    );
                }
            }
            _ => {}
        }
    }
}
