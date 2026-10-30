package com.tobiilinux.poc

import android.Manifest
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

private const val VID = 0x2104
private const val PID = 0x0313
private const val ACTION_USB_PERMISSION = "com.tobiilinux.poc.USB_PERMISSION"

class MainActivity : AppCompatActivity() {
    private lateinit var usbManager: UsbManager
    private lateinit var statusText: TextView
    private lateinit var gazeText: TextView
    private lateinit var calibrateButton: Button
    private lateinit var gazeDotView: GazeDotView
    private var connection: android.hardware.usb.UsbDeviceConnection? = null
    private val poll = Handler(Looper.getMainLooper())

    private val calibration = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        statusText.text = if (result.resultCode == RESULT_OK) "Running (calibrated)" else "Running"
    }

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            findDeviceAndRequest()
        } else {
            statusText.text = "Camera permission refused (required by Android for this USB device's video-class interface, even though this app does not use it)"
        }
    }

    private val permissionReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != ACTION_USB_PERMISSION) return
            val device: UsbDevice? = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
            val granted = intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)
            if (granted && device != null) {
                openAndStart(device)
            } else {
                statusText.text = "USB permission refused"
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        statusText = findViewById(R.id.statusText)
        gazeText = findViewById(R.id.gazeText)
        calibrateButton = findViewById(R.id.calibrateButton)
        gazeDotView = findViewById(R.id.gazeDotView)
        calibrateButton.setOnClickListener {
            calibration.launch(Intent(this, CalibrationActivity::class.java))
        }
        usbManager = getSystemService(Context.USB_SERVICE) as UsbManager

        val filter = IntentFilter(ACTION_USB_PERMISSION)
        if (Build.VERSION.SDK_INT >= 33) {
            registerReceiver(permissionReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            @Suppress("UnspecifiedRegisterReceiverFlag")
            registerReceiver(permissionReceiver, filter)
        }

        requestCameraThenDevice()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        if (intent.action == UsbManager.ACTION_USB_DEVICE_ATTACHED) {
            requestCameraThenDevice()
        }
    }

    /**
     * Android refuses USB permission for this device without a prior grant
     * of the Camera runtime permission: it advertises a USB Video Class
     * interface (the tracker's own "Hello sensor") alongside the vendor
     * interface this app actually talks to, and Android gates the whole
     * device on that account, with no USB permission dialog shown at all
     * until Camera is granted (confirmed in logcat,
     * `UsbUserPermissionManager`). This app never opens that interface.
     */
    private fun requestCameraThenDevice() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            findDeviceAndRequest()
        } else {
            statusText.text = "Requesting Camera permission (required by Android for this USB device)..."
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun findDeviceAndRequest() {
        val device = usbManager.deviceList.values.firstOrNull {
            it.vendorId == VID && it.productId == PID
        }
        if (device == null) {
            statusText.text = "Plug in the Tobii Eye Tracker 5"
            return
        }
        if (usbManager.hasPermission(device)) {
            openAndStart(device)
            return
        }
        val flags = if (Build.VERSION.SDK_INT >= 31) PendingIntent.FLAG_IMMUTABLE else 0
        val pendingIntent = PendingIntent.getBroadcast(
            this, 0, Intent(ACTION_USB_PERMISSION), flags
        )
        statusText.text = "Requesting USB permission..."
        usbManager.requestPermission(device, pendingIntent)
    }

    private fun openAndStart(device: UsbDevice) {
        val conn = usbManager.openDevice(device)
        if (conn == null) {
            statusText.text = "Could not open the device"
            return
        }
        connection = conn
        // Not claiming interface 0 here: the native side claims it itself
        // (over the wrapped fd, through libusb) once the engine starts.
        statusText.text = "Starting engine..."
        NativeBridge.startEngine(conn.fileDescriptor)
        statusText.text = "Running"
        calibrateButton.visibility = android.view.View.VISIBLE
        poll.post(pollTick)
    }

    private val pollTick = object : Runnable {
        override fun run() {
            val g = NativeBridge.latestGaze()
            val valid = g[2] != 0.0
            gazeText.text = "x=%.3f  y=%.3f  valid=%s".format(g[0], g[1], valid)
            gazeDotView.point = floatArrayOf(g[0].toFloat(), g[1].toFloat())
            gazeDotView.valid = valid
            gazeDotView.invalidate()
            poll.postDelayed(this, 33)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        poll.removeCallbacksAndMessages(null)
        NativeBridge.stopEngine()
        connection?.close()
        unregisterReceiver(permissionReceiver)
    }
}
