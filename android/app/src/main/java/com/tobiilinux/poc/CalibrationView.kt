package com.tobiilinux.poc

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Handler
import android.os.Looper
import android.util.AttributeSet
import android.view.View

/**
 * Draws one target dot at a time and walks it through [points] (normalised
 * display coordinates): a travel animation to the next point, a dwell to
 * let the user's gaze settle, then a shrink while [onCollect] runs — off
 * the main thread, since a collect blocks for under a second — before
 * moving on. Calls [onDone] once every point has been collected.
 *
 * The two-phase split (travel then dwell) matters: collecting while the dot
 * is still moving would sample a gaze that is still catching up to it.
 */
class CalibrationView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    /** One point's collection: normalised `(x, y)`; `true` on success. */
    var onCollect: (Float, Float) -> Boolean = { _, _ -> false }

    /** The whole sequence finished. */
    var onDone: () -> Unit = {}

    var points: List<FloatArray> = emptyList()

    private val handler = Handler(Looper.getMainLooper())
    private var index = -1
    private var from = floatArrayOf(0.5f, 0.5f)
    private var to = floatArrayOf(0.5f, 0.5f)
    private var current = floatArrayOf(0.5f, 0.5f)
    private var radiusFraction = 1f
    private var phase = Phase.IDLE

    private enum class Phase { IDLE, TRAVEL, DWELL, COLLECT }

    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    companion object {
        private const val TRAVEL_MS = 500L
        private const val DWELL_MS = 450L
        // However long the device actually takes to answer a collect (it
        // can refuse almost at once instead of sampling for the usual
        // ~0.6-0.9 s), the dot holds the "capturing" phase for at least
        // this long: otherwise a fast refusal shortens that one point's
        // pacing and the sequence visibly speeds up partway through.
        private const val MIN_COLLECT_MS = 700L
        private const val FRAME_MS = 16L
        private const val BASE_RADIUS_PX = 28f
    }

    /** Starts the sequence from its first point. */
    fun start() {
        index = -1
        advance()
    }

    private fun advance() {
        index++
        if (index >= points.size) {
            phase = Phase.IDLE
            onDone()
            return
        }
        from = current
        to = points[index]
        phase = Phase.TRAVEL
        animateTravel(System.currentTimeMillis())
    }

    private fun animateTravel(startedAt: Long) {
        val t = ((System.currentTimeMillis() - startedAt).toFloat() / TRAVEL_MS).coerceIn(0f, 1f)
        current = floatArrayOf(from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t)
        radiusFraction = 1f
        invalidate()
        if (t < 1f) {
            handler.postDelayed({ animateTravel(startedAt) }, FRAME_MS)
        } else {
            phase = Phase.DWELL
            handler.postDelayed({ startCollect() }, DWELL_MS)
        }
    }

    private fun startCollect() {
        phase = Phase.COLLECT
        radiusFraction = 0.4f
        invalidate()
        val (x, y) = to[0] to to[1]
        val startedAt = System.currentTimeMillis()
        Thread {
            onCollect(x, y)
            val remaining = MIN_COLLECT_MS - (System.currentTimeMillis() - startedAt)
            if (remaining > 0) {
                Thread.sleep(remaining)
            }
            handler.post { advance() }
        }.start()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (phase == Phase.IDLE) return
        val cx = current[0] * width
        val cy = current[1] * height
        val radius = BASE_RADIUS_PX * radiusFraction
        canvas.drawCircle(cx, cy, radius, dotPaint)
        canvas.drawCircle(cx, cy, BASE_RADIUS_PX + 6f, ringPaint)
    }
}
