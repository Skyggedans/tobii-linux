# tobii-proto test fixtures

Single messages captured from the Windows Stream Engine 4.1.0.3 talking to a
Tobii Eye Tracker 5, one per file as a line of hex. They are what the tests
compare the decoders and command builders against byte for byte.

The device serial number has been replaced by `IS50F-000000000000` (same
length); nothing else is modified.

| file | capture | what it is |
|---|---|---|
| `init-rsp-<cmd>.hex` | init.pcapng | the device's response to init command `<cmd>`: 1000 protocol version, 1200 stream catalogue, 1330 properties, 1400 track box, 1420 identity strings, 1430 display area, 1490 status strings, 1650/1670 output rate, 2110 mounting, 3170 state |
| `init-rsp-2120.hex` | init.pcapng frame 773 | the hardware configuration (post-calib-init.pcapng frame 223 is byte-identical); on Linux the ET5 answers 2120 with the header alone |
| `init-notify-3180.hex` | init.pcapng | the one notification during init |
| `init-presence.hex` | init.pcapng | the first 0x504 presence message (present) |
| `change-display-cmd-1440.hex` | change-display.pcapng | the host setting a 597x336 mm display at runtime |
| `change-display-rsp-1440.hex` | change-display.pcapng | the device's empty ack |
| `change-display-notify-1450.hex` | change-display.pcapng | the display-area notification that follows |
| `calib-cmd-<cmd>-seq<n>.hex` | calibration.pcapng | calibration commands: 1010 start, 1020 stop, 1030 collect, 1060 clear, 1070 compute, 1100 read |
| `calib-notify-3220-frame<n>.hex` | calibration.pcapng | calibration-id notifications |
| `calib-rsp-1100-head.hex` | calibration.pcapng | the first message of a chunked calibration read |
| `session1-gaze-frame.hex` | session1.pcapng frame 467 | 0x500 gaze frame 43780; session1.jsonl has what the Stream Engine delivered for it |
| `session2-presence-away.hex` | session2.pcapng frame 8391 | a 0x504 presence message (away) |

To extract more: `tshark -r <capture> -Y usb.capdata -T fields -e frame.number
-e usb.endpoint_address -e usb.capdata`; device->host messages start with
`01000000`, host->device commands with `00000000`.
