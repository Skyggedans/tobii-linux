//! Small fixed-size linear algebra and calibration helpers shared by the
//! `OpenTrack` pose pipeline and the offline analysis: rest-origin averaging,
//! angle wrapping, 3-vector/3x3 products and a tiny Gaussian solver.

/// Rest-origin calibration for a 3-vector: once `origin` is set it is returned
/// as-is; otherwise, while `can_calibrate`, `value` is accumulated into
/// `sum`/`count` and the mean becomes the origin after `samples` values.
/// Returns `None` until calibrated.
pub(crate) fn calibrate_origin(
    origin: &mut Option<[f64; 3]>,
    sum: &mut [f64; 3],
    count: &mut usize,
    samples: usize,
    value: [f64; 3],
    can_calibrate: bool,
) -> Option<[f64; 3]> {
    if let Some(origin) = *origin {
        return Some(origin);
    }

    if !can_calibrate {
        return None;
    }

    for (s, v) in sum.iter_mut().zip(&value) {
        *s += v;
    }
    *count += 1;

    if *count < samples {
        return None;
    }

    let calibrated = [
        sum[0] / *count as f64,
        sum[1] / *count as f64,
        sum[2] / *count as f64,
    ];
    *origin = Some(calibrated);
    Some(calibrated)
}

/// Scalar counterpart of [`calibrate_origin`].
pub(crate) fn calibrate_scalar_origin(
    origin: &mut Option<f64>,
    sum: &mut f64,
    count: &mut usize,
    samples: usize,
    value: f64,
    can_calibrate: bool,
) -> Option<f64> {
    if let Some(origin) = *origin {
        return Some(origin);
    }

    if !can_calibrate {
        return None;
    }

    *sum += value;
    *count += 1;

    if *count < samples {
        return None;
    }

    let calibrated = *sum / *count as f64;
    *origin = Some(calibrated);
    Some(calibrated)
}

/// Wrap an angle in degrees into `(-180, 180]`.
#[must_use]
pub(crate) fn normalize_angle_deg(mut angle: f64) -> f64 {
    while angle > 180.0 {
        angle -= 360.0;
    }
    while angle < -180.0 {
        angle += 360.0;
    }
    angle
}

/// Dot product of two 3-vectors.
#[must_use]
pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean length of a 3-vector.
#[must_use]
pub(crate) fn norm3(value: [f64; 3]) -> f64 {
    dot3(value, value).sqrt()
}

/// Multiply every element of a 3x3 matrix by `scale`.
#[must_use]
pub(crate) fn scale_matrix(mut matrix: [[f64; 3]; 3], scale: f64) -> [[f64; 3]; 3] {
    for row in &mut matrix {
        for value in row {
            *value *= scale;
        }
    }
    matrix
}

/// Pick between the "rotation" hypothesis (translation explained by the
/// rotation coupling) and the "translation" hypothesis (angles explained by
/// the translation coupling) for one raw pose. Returns `(translation, angles)`.
#[must_use]
pub(crate) fn choose_pose_hypothesis(
    raw_translation: [f64; 3],
    raw_angles: [f64; 3],
    rotation_comp: [[f64; 3]; 3],
    angle_translation_comp: [[f64; 3]; 3],
    translation_deadzone: f64,
) -> ([f64; 3], [f64; 3]) {
    let rotation_translation = [
        raw_translation[0] - dot3(rotation_comp[0], raw_angles),
        raw_translation[1] - dot3(rotation_comp[1], raw_angles),
        raw_translation[2] - dot3(rotation_comp[2], raw_angles),
    ];
    let translation_angles =
        angles_from_translation(raw_angles, raw_translation, angle_translation_comp);

    let raw_translation_len = dot3(raw_translation, raw_translation).sqrt();
    let rotation_translation_len = dot3(rotation_translation, rotation_translation).sqrt();

    if rotation_translation_len <= translation_deadzone
        || rotation_translation_len <= raw_translation_len * 0.45
    {
        (rotation_translation, raw_angles)
    } else if raw_translation_len > translation_deadzone {
        (raw_translation, translation_angles)
    } else {
        (rotation_translation, raw_angles)
    }
}

/// Angles with the translation-induced coupling removed:
/// `raw - angle_translation_comp * translation`.
#[must_use]
pub(crate) fn angles_from_translation(
    raw_angles: [f64; 3],
    translation: [f64; 3],
    angle_translation_comp: [[f64; 3]; 3],
) -> [f64; 3] {
    [
        raw_angles[0] - dot3(angle_translation_comp[0], translation),
        raw_angles[1] - dot3(angle_translation_comp[1], translation),
        raw_angles[2] - dot3(angle_translation_comp[2], translation),
    ]
}

/// Yaw (deg) of a direction vector: rotation about +y, from +z toward +x.
#[must_use]
pub(crate) fn vector_yaw_deg(vec: [f64; 3]) -> f64 {
    vec[0].atan2(vec[2]).to_degrees()
}

/// Pitch (deg) of a direction vector: elevation above the xz plane (image y
/// points down, so `-y` is up).
#[must_use]
pub(crate) fn vector_pitch_deg(vec: [f64; 3]) -> f64 {
    (-vec[1])
        .atan2((vec[0] * vec[0] + vec[2] * vec[2]).sqrt())
        .to_degrees()
}

/// Roll (deg) of a vector projected onto the image xy plane.
#[must_use]
pub(crate) fn vector_roll_xy_deg(vec: [f64; 3]) -> f64 {
    vec[1].atan2(vec[0]).to_degrees()
}

/// Solve `a * x = b` by Gauss-Jordan elimination with partial pivoting and a
/// tiny diagonal ridge. Returns `None` when a pivot is (numerically) zero.
#[must_use]
pub(crate) fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for i in 0..3 {
        a[i][i] += 1e-9;
        let mut pivot = i;
        for row in i + 1..3 {
            if a[row][i].abs() > a[pivot][i].abs() {
                pivot = row;
            }
        }
        if a[pivot][i].abs() < 1e-12 {
            return None;
        }
        a.swap(i, pivot);
        b.swap(i, pivot);

        let div = a[i][i];
        for value in a[i].iter_mut().skip(i) {
            *value /= div;
        }
        b[i] /= div;

        let row_i = a[i];
        let b_i = b[i];
        for (row, (a_row, b_row)) in a.iter_mut().zip(&mut b).enumerate() {
            if row == i {
                continue;
            }
            let factor = a_row[i];
            for (value, pivot_value) in a_row.iter_mut().zip(&row_i).skip(i) {
                *value -= factor * pivot_value;
            }
            *b_row -= factor * b_i;
        }
    }

    Some(b)
}
