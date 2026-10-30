package com.tobiilinux.poc

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/**
 * An always-on-top dot at the latest gaze point (normalised display
 * coordinates), so the calibration's effect is visible at a glance and not
 * just in the `x=…  y=…` text. Hidden while [point] is `null` (nothing
 * gazed at yet) or [valid] is `false` (the device lost the face): a dot
 * frozen at the last valid spot would be misleading. Never intercepts
 * touches, so it can sit over the rest of the screen, buttons included.
 */
class GazeDotView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    var point: FloatArray? = null
    var valid: Boolean = false

    init {
        isClickable = false
        isFocusable = false
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(200, 0, 230, 120)
    }
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    companion object {
        private const val RADIUS_PX = 18f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val p = point ?: return
        if (!valid) return
        val cx = (p[0] * width).coerceIn(0f, width.toFloat())
        val cy = (p[1] * height).coerceIn(0f, height.toFloat())
        canvas.drawCircle(cx, cy, RADIUS_PX, fillPaint)
        canvas.drawCircle(cx, cy, RADIUS_PX, ringPaint)
    }
}
