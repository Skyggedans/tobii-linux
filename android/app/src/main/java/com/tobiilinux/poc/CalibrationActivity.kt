package com.tobiilinux.poc

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * The 7-point pattern the Windows engine calibrates with
 * (`tobii_calib::blob::STIMULUS_POINTS`), run twice: the device keeps the
 * last 14 points (two rounds), and two rounds replace every point of
 * whatever calibration was active before.
 */
private val STIMULUS_POINTS = listOf(
    floatArrayOf(0.5f, 0.5f),
    floatArrayOf(0.5f, 0.1f),
    floatArrayOf(0.9f, 0.9f),
    floatArrayOf(0.1f, 0.9f),
    floatArrayOf(0.1f, 0.1f),
    floatArrayOf(0.9f, 0.1f),
    floatArrayOf(0.5f, 0.9f),
)
private val SEQUENCE = STIMULUS_POINTS + STIMULUS_POINTS

class CalibrationActivity : AppCompatActivity() {
    private lateinit var view: CalibrationView
    private lateinit var resultText: TextView
    private val handler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_calibration)
        view = findViewById(R.id.calibrationView)
        resultText = findViewById(R.id.resultText)
        val resultButtons = findViewById<android.widget.LinearLayout>(R.id.resultButtons)
        val keepButton = findViewById<android.widget.Button>(R.id.keepButton)
        val discardButton = findViewById<android.widget.Button>(R.id.discardButton)

        resultText.text = "Starting..."
        resultText.visibility = android.view.View.VISIBLE
        Thread {
            val started = NativeBridge.calibrationStart(filesDir.absolutePath)
            handler.post {
                if (started) {
                    resultText.visibility = android.view.View.GONE
                    view.points = SEQUENCE
                    view.onCollect = { x, y -> NativeBridge.calibrationCollect(x.toDouble(), y.toDouble()) }
                    view.onDone = { onSequenceDone(resultText, resultButtons, keepButton, discardButton) }
                    view.start()
                } else {
                    resultText.text = "Could not start calibration: ${NativeBridge.lastCalibrationError()}"
                    showCloseButton(resultButtons, keepButton, discardButton)
                }
            }
        }.start()
    }

    private fun onSequenceDone(
        resultText: TextView,
        resultButtons: android.widget.LinearLayout,
        keepButton: android.widget.Button,
        discardButton: android.widget.Button,
    ) {
        resultText.text = "Computing..."
        resultText.visibility = android.view.View.VISIBLE
        Thread {
            val id = NativeBridge.calibrationCompute()
            handler.post {
                if (id >= 0) {
                    resultText.text = "Calibration computed (id=$id). Keep it?"
                    resultButtons.visibility = android.view.View.VISIBLE
                    keepButton.setOnClickListener { finishSession(resultText, resultButtons, keep = true) }
                    discardButton.setOnClickListener { finishSession(resultText, resultButtons, keep = false) }
                } else {
                    resultText.text = "Calibration failed: ${NativeBridge.lastCalibrationError()}"
                    // Still ends the device's session (keep = false, so the
                    // previous calibration goes back), but waits for the
                    // user to read the error instead of finishing at once.
                    Thread { NativeBridge.calibrationStop(false) }.start()
                    showCloseButton(resultButtons, keepButton, discardButton)
                }
            }
        }.start()
    }

    /** One "Close" button in place of Keep/Discard, for an error message. */
    private fun showCloseButton(
        resultButtons: android.widget.LinearLayout,
        keepButton: android.widget.Button,
        discardButton: android.widget.Button,
    ) {
        keepButton.visibility = android.view.View.GONE
        discardButton.text = "Close"
        discardButton.setOnClickListener {
            setResult(RESULT_CANCELED)
            finish()
        }
        resultButtons.visibility = android.view.View.VISIBLE
    }

    private fun finishSession(
        resultText: TextView,
        resultButtons: android.widget.LinearLayout,
        keep: Boolean,
    ) {
        resultButtons.visibility = android.view.View.GONE
        resultText.text = if (keep) "Saving..." else "Discarding..."
        Thread {
            val saved = NativeBridge.calibrationStop(keep)
            handler.post {
                setResult(if (saved) RESULT_OK else RESULT_CANCELED)
                finish()
            }
        }.start()
    }
}
