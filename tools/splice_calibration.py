#!/usr/bin/env python3
"""Splice a Tobii calibration out of a USB capture into init_packets_ep.txt.

The device init replay carries a ~660 KB config blob; the per-user gaze
calibration lives as a small float table (target<->measured gaze) in the blob's
tail, plus a 0x31f41 reference-geometry command packet. This tool transplants
those from a Windows capture into our base init so the Linux daemon runs the
user's real calibration.

  --table SRC.pcapng   take the dense gaze table (blob tail) from SRC
  --ref   SRC.pcapng   take the 0x31f41 reference packet from SRC
  --both  SRC.pcapng   shorthand for --table SRC --ref SRC

Examples:
  ./splice_calibration.py --table calibration.pcapng --out init_packets_calib.txt
  ./splice_calibration.py --ref post-calib-init.pcapng --out init_packets_ref.txt
"""
import argparse
import subprocess
import sys

# The dense gaze table lives in the final blob chunk; copy the whole last chunk
# (the bytes before the table are identical template, so this is safe).
TAIL_BYTES = 1092
HEADER_VERSION_OFF = 15  # blob byte that bumps on recalibration


def tshark_05_packets(pcapng):
    """Ordered list of (frame, hex) for EP 0x05 OUT data packets."""
    out = subprocess.run(
        ["tshark", "-r", pcapng, "-Y",
         "usb.endpoint_address==0x05 && usb.data_len>0",
         "-T", "fields", "-e", "frame.number", "-e", "usb.capdata"],
        capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[1]:
            rows.append((int(p[0]), p[1].replace(":", "")))
    rows.sort()
    return [h for _, h in rows]


def blob_from_packets(packets):
    """Concatenate the big (>1000 byte) chunks into the config blob."""
    return bytes.fromhex("".join(h for h in packets if len(h) // 2 > 1000))


def read_init(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ep, hexd = line.split()
        out.append([ep, hexd.replace(":", "")])
    return out


def find_ref_packet(packets):
    """The 0x31f41 reference command: small, contains 00031f41 and the 00003039
    trailer constant."""
    for i, h in enumerate(packets):
        if len(h) // 2 <= 2000 and "00031f41" in h and "00003039" in h:
            return i, h
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="init_packets_ep.txt")
    ap.add_argument("--table", metavar="SRC.pcapng")
    ap.add_argument("--ref", metavar="SRC.pcapng")
    ap.add_argument("--both", metavar="SRC.pcapng")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.both:
        a.table = a.ref = a.both
    if not (a.table or a.ref):
        ap.error("need --table, --ref or --both")

    base = read_init(a.base)
    chunk_idx = [i for i, (_, h) in enumerate(base) if len(h) // 2 > 1000]
    base_blob = bytearray(blob_from_packets([h for _, h in base]))
    print(f"base: {len(base)} packets, blob {len(base_blob)} bytes "
          f"in {len(chunk_idx)} chunks")

    if a.table:
        src_blob = blob_from_packets(tshark_05_packets(a.table))
        if len(src_blob) != len(base_blob):
            sys.exit(f"blob size mismatch: src {len(src_blob)} vs base {len(base_blob)}")
        before = bytes(base_blob)
        base_blob[HEADER_VERSION_OFF] = src_blob[HEADER_VERSION_OFF]
        base_blob[-TAIL_BYTES:] = src_blob[-TAIL_BYTES:]
        changed = sum(1 for x, y in zip(before, base_blob) if x != y)
        print(f"--table: spliced gaze table from {a.table} ({changed} bytes changed)")
        # write the (only) modified chunks back
        pos = 0
        for i in chunk_idx:
            ln = len(base[i][1]) // 2
            base[i][1] = bytes(base_blob[pos:pos + ln]).hex()
            pos += ln

    if a.ref:
        src_pkts = tshark_05_packets(a.ref)
        si, sref = find_ref_packet(src_pkts)
        bi, bref = find_ref_packet([h for _, h in base])
        if sref is None or bref is None:
            sys.exit("could not locate the 0x31f41 reference packet")
        base[bi][1] = sref
        print(f"--ref: spliced 0x31f41 reference from {a.ref}")

    with open(a.out, "w") as f:
        for ep, h in base:
            f.write(f"{ep} {h}\n")
    print(f"wrote {a.out}  ({len(base)} packets)")


if __name__ == "__main__":
    main()
