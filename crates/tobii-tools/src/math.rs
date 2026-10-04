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

/// Most sweeps [`symmetric_eigen`] makes. Its rotations converge
/// quadratically: a 4x4 takes a handful.
const JACOBI_MAX_SWEEPS: usize = 64;

/// The eigenvalues and eigenvectors of the symmetric matrix `a`, by cyclic
/// Jacobi rotations: `(values, vectors)`, the eigenvector of `values[k]`
/// being column `k` of `vectors` (`vectors[i][k]` for each `i`), in no
/// particular order. The vectors are orthonormal. The rotations go on until
/// what is left off the diagonal is within `f64::EPSILON` of the matrix's
/// norm, so the eigenvalues are as exact as the matrix and each eigenvector
/// as exact as the gap to the next eigenvalue allows.
///
/// `None` for a matrix with an element that is not finite, or should the
/// rotations not converge within [`JACOBI_MAX_SWEEPS`] sweeps.
#[must_use]
pub(crate) fn symmetric_eigen<const N: usize>(
    mut a: [[f64; N]; N],
) -> Option<([f64; N], [[f64; N]; N])> {
    let mut vectors: [[f64; N]; N] =
        std::array::from_fn(|i| std::array::from_fn(|j| f64::from(u8::from(i == j))));
    let norm_sq: f64 = a.iter().flatten().map(|v| v * v).sum();
    if !norm_sq.is_finite() {
        return None;
    }
    let tolerance = f64::EPSILON * f64::EPSILON * norm_sq;
    for _ in 0..JACOBI_MAX_SWEEPS {
        let off: f64 = a
            .iter()
            .enumerate()
            .flat_map(|(p, row)| row.iter().skip(p + 1))
            .map(|v| v * v)
            .sum();
        if off <= tolerance {
            return Some((std::array::from_fn(|k| a[k][k]), vectors));
        }
        for p in 0..N {
            for q in p + 1..N {
                let apq = a[p][q];
                if apq == 0.0 {
                    continue;
                }
                // The rotation J (c and s at (p, p), (p, q); -s and c at
                // (q, p), (q, q)) that zeroes a[p][q] in J^T A J, of the
                // smaller angle (Golub and Van Loan, sym.schur2).
                let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = tau.signum() / (tau.abs() + tau.hypot(1.0));
                let c = 1.0 / t.hypot(1.0);
                let s = t * c;
                for row in &mut a {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
                let (row_p, row_q) = (a[p], a[q]);
                a[p] = std::array::from_fn(|k| c * row_p[k] - s * row_q[k]);
                a[q] = std::array::from_fn(|k| s * row_p[k] + c * row_q[k]);
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for row in &mut vectors {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `a b`.
    fn matmul<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..N).map(|k| a[i][k] * b[k][j]).sum()))
    }

    /// `aᵀ`.
    fn transpose<const N: usize>(a: &[[f64; N]; N]) -> [[f64; N]; N] {
        std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
    }

    /// An orthonormal 4x4: the rotation of the unit quaternion `q`
    /// (w, x, y, z) as a left-multiplication matrix.
    fn orthonormal(q: [f64; 4]) -> [[f64; 4]; 4] {
        let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        let [w, x, y, z] = q.map(|v| v / n);
        [[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]]
    }

    /// `values` and `vectors` decompose `a`: `a v = λ v` for each pair, the
    /// vectors orthonormal, and the values `want` in some order.
    fn assert_decomposes<const N: usize>(
        a: &[[f64; N]; N],
        (values, vectors): &([f64; N], [[f64; N]; N]),
        want: &[f64; N],
    ) {
        let scale = a.iter().flatten().map(|v| v.abs()).fold(1.0, f64::max);
        for (k, value) in values.iter().enumerate() {
            let v: [f64; N] = std::array::from_fn(|i| vectors[i][k]);
            for (row, vi) in a.iter().zip(&v) {
                let av: f64 = row.iter().zip(&v).map(|(x, y)| x * y).sum();
                assert!((av - value * vi).abs() <= 1e-13 * scale, "{a:?}: λ {value}");
            }
        }
        let gram = matmul(&transpose(vectors), vectors);
        for (i, row) in gram.iter().enumerate() {
            for (j, g) in row.iter().enumerate() {
                assert!((g - f64::from(u8::from(i == j))).abs() <= 1e-14, "{gram:?}");
            }
        }
        let mut got = *values;
        got.sort_by(f64::total_cmp);
        let mut want = *want;
        want.sort_by(f64::total_cmp);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-13 * scale, "{got:?} != {want:?}");
        }
    }

    #[test]
    fn a_symmetric_matrix_decomposes_into_its_eigenvalues_and_vectors() {
        // V D Vᵀ of a known D: distinct values of both signs; a repeated one;
        // two that differ by one part in 10^9.
        let v = orthonormal([0.9, -0.3, 0.2, 0.25]);
        for d in [
            [4.0, -1.5, 0.25, 9.0],
            [5.0, 5.0, -1.0, 2.0],
            [1.0, 1.0 + 1e-9, -3.0, 0.5],
        ] {
            let diag: [[f64; 4]; 4] =
                std::array::from_fn(|i| std::array::from_fn(|j| if i == j { d[i] } else { 0.0 }));
            let a = matmul(&matmul(&v, &diag), &transpose(&v));
            let a: [[f64; 4]; 4] =
                std::array::from_fn(|i| std::array::from_fn(|j| 0.5 * (a[i][j] + a[j][i])));
            let eigen = symmetric_eigen(a).expect("converges");
            assert_decomposes(&a, &eigen, &d);
        }
        // A 3x3, a diagonal one, the zero matrix.
        let a = [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]];
        let root2 = 2f64.sqrt();
        let eigen = symmetric_eigen(a).expect("converges");
        assert_decomposes(&a, &eigen, &[2.0 - root2, 2.0, 2.0 + root2]);
        let diagonal = [[3.0, 0.0], [0.0, -7.0]];
        assert_eq!(
            symmetric_eigen(diagonal),
            Some(([3.0, -7.0], [[1.0, 0.0], [0.0, 1.0]]))
        );
        assert_eq!(
            symmetric_eigen([[0.0; 2]; 2]),
            Some(([0.0; 2], [[1.0, 0.0], [0.0, 1.0]]))
        );
    }

    #[test]
    fn a_matrix_with_an_element_that_is_not_finite_has_no_decomposition() {
        for bad in [f64::NAN, f64::INFINITY] {
            let a = [[1.0, bad], [bad, 2.0]];
            assert_eq!(symmetric_eigen(a), None);
        }
    }
}
