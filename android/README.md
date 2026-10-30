# Android

A minimal proof of concept: the Tobii Eye Tracker 5's gaze stream in an
Android app, no root, over the file descriptor `UsbManager` hands out (see
`crates/tobii-android`). One engine runs inside the app's own process — not
the daemon/client split the Linux and macOS builds use — and a calibration
screen runs the same captured command sequence as the daemon's own
(`tobiid::calibration`) directly against the engine's `Commands` handle.

## Building

1. Build the native library for your device's ABI (`arm64-v8a` for a current
   phone or tablet):

   ```sh
   cargo ndk -t arm64-v8a build -p tobii-android --release
   ```

   This excludes head-pose/ONNX Runtime (`tobii-usb`'s `head-pose` feature,
   on by default elsewhere): its prebuilt Android binaries don't link
   cleanly against the NDK's `libc++_shared`, and gaze itself (stream
   `0x500`) never needs them.

2. Copy the built library where Gradle picks it up without further
   configuration:

   ```sh
   cp ../target/aarch64-linux-android/release/libtobii_android.so \
     app/src/main/jniLibs/arm64-v8a/
   ```

3. Build the APK (needs a JDK 17+ on `JAVA_HOME`):

   ```sh
   gradle assembleDebug
   ```

## Running

The tracker exposes a USB Video Class interface alongside the vendor one
this app talks to; Android gates USB permission for the whole device on
that account, so the app also requests the `CAMERA` runtime permission
(never opened) before `UsbManager` will show its own dialog — see
`MainActivity.requestCameraThenDevice`.
