//! Export of the face fits `image83-replay` makes of a recorded log's 0x50e
//! images: one [`FaceFit`] per image from the production tracker, at full
//! precision, for fitting the head pose model's constants offline and for
//! comparing the tracker with the Python reference it was ported from.
//!
//! `--fits` is a CSV with one row per decoded 0x50e image, in log order (the
//! rows of `--csv`):
//!
//! - `image_idx`: the image's index among the log's decoded 0x50e images,
//!   from 0; `device_ts_us`: its device timestamp (µs).
//! - `face`: 1 when the tracker found a face, else 0. When 0, every later
//!   field of the row is empty.
//! - `score`: the landmark model's face-presence logit (never negative), the
//!   model's f32 written as the f64 of the same value; `found_by_detector`: 1
//!   when the face detector placed the crop the face was found in, else 0.
//! - `r_cam_00` .. `r_cam_22`: the rotation of the canonical face mesh into
//!   the camera frame (object -> camera; camera x right, y down, z forward),
//!   row-major: `r_cam_ij` is row i, column j of [`FaceFit::rotation`].
//! - `t_cam_x_mm`, `t_cam_y_mm`, `t_cam_z_mm`: the mesh origin in the camera
//!   frame, mm ([`FaceFit::translation_mm`]).
//! - Six points of the 280-px image, each as `<name>_u`, `<name>_v`, in
//!   continuous pixels (x right, y down, pixel k spanning `[k, k + 1)`):
//!   `centroid`, the mean of the 468 landmarks; `nose_tip`, landmark 1;
//!   `eye_image_left` and `eye_image_right`, the means of the 16 contour
//!   landmarks of the eye on the image's left (the subject's right eye:
//!   [`EYE_CONTOUR_IMAGE_LEFT`]) and on its right
//!   ([`EYE_CONTOUR_IMAGE_RIGHT`]); `corner_33` and `corner_263`, the eyes'
//!   outer corners.
//!
//! Every number is written in the shortest form that reads back as the same
//! f64 (Rust's `Display`, never an exponent; `NaN`, `inf` and `-inf` as
//! such).
//!
//! `--landmarks` is raw little-endian f32 with no header: for every image, in
//! log order like the CSV's rows, its 468 landmarks as (u, v) pairs in
//! landmark order, the CSV's continuous pixels rounded to f32; 468 x 2 x 4 =
//! 3744 bytes per image, all 936 values NaN when the image has no face.
//! `NumPy` reads it as `np.fromfile(path, '<f4').reshape(-1, 468, 2)`.

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use tobii_pose::track::FaceFit;

/// Landmarks of one fit: the face-landmark model's 468, as
/// [`FaceFit::landmarks`] has them.
const LANDMARKS: usize = 468;

/// The nose tip landmark.
const NOSE_TIP: usize = 1;

/// The 16 landmarks of the contour of the eye on the image's left, the
/// subject's right eye: the head pose study's eye-position model takes the
/// eye at their mean (`landmarks.contour_image_left` of its
/// `final_params.json`).
const EYE_CONTOUR_IMAGE_LEFT: [usize; 16] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
];

/// The 16 landmarks of the contour of the eye on the image's right, the
/// subject's left eye (`landmarks.contour_image_right`).
const EYE_CONTOUR_IMAGE_RIGHT: [usize; 16] = [
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
];

/// The eyes' outer corners, on the image's left and right: the landmarks the
/// tracker turns its crop by.
const EYE_CORNERS: [usize; 2] = [33, 263];

/// The `--fits` CSV's header (see the module docs).
const HEADER: &str = concat!(
    "image_idx,device_ts_us,face,score,found_by_detector,",
    "r_cam_00,r_cam_01,r_cam_02,r_cam_10,r_cam_11,r_cam_12,r_cam_20,r_cam_21,r_cam_22,",
    "t_cam_x_mm,t_cam_y_mm,t_cam_z_mm,",
    "centroid_u,centroid_v,nose_tip_u,nose_tip_v,",
    "eye_image_left_u,eye_image_left_v,eye_image_right_u,eye_image_right_v,",
    "corner_33_u,corner_33_v,corner_263_u,corner_263_v",
);

/// Fields of a row after `face`: the score, `found_by_detector`, the
/// rotation, the translation and six points.
const FIT_FIELDS: usize = 2 + 9 + 3 + 2 * 6;

/// Bytes of one image's record in the `--landmarks` file.
const LANDMARK_RECORD_BYTES: usize = LANDMARKS * 2 * size_of::<f32>();

/// What the export writes of one image's [`FaceFit`]. Only tobii-pose builds
/// a `FaceFit`, so the tests build these instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FitFields<'a> {
    /// [`FaceFit::rotation`].
    rotation: [[f64; 3]; 3],
    /// [`FaceFit::translation_mm`].
    translation_mm: [f64; 3],
    /// [`FaceFit::landmarks`].
    landmarks: &'a [[f64; 2]; LANDMARKS],
    /// [`FaceFit::score`].
    score: f32,
    /// [`FaceFit::found_by_detector`].
    found_by_detector: bool,
}

impl<'a> From<&FaceFit<'a>> for FitFields<'a> {
    fn from(fit: &FaceFit<'a>) -> Self {
        Self {
            rotation: fit.rotation,
            translation_mm: fit.translation_mm,
            landmarks: fit.landmarks,
            score: fit.score,
            found_by_detector: fit.found_by_detector,
        }
    }
}

/// The six points of a fit's landmarks the CSV carries, in its column order:
/// the centroid of all of them, the nose tip, the centroids of the eye
/// contours on the image's left and right, and the two outer eye corners.
#[must_use]
fn image_points(landmarks: &[[f64; 2]; LANDMARKS]) -> [[f64; 2]; 6] {
    [
        centroid(landmarks.iter()),
        landmarks[NOSE_TIP],
        centroid(EYE_CONTOUR_IMAGE_LEFT.iter().map(|&k| &landmarks[k])),
        centroid(EYE_CONTOUR_IMAGE_RIGHT.iter().map(|&k| &landmarks[k])),
        landmarks[EYE_CORNERS[0]],
        landmarks[EYE_CORNERS[1]],
    ]
}

/// The mean of `points`, summed in order.
#[must_use]
fn centroid<'p>(points: impl ExactSizeIterator<Item = &'p [f64; 2]>) -> [f64; 2] {
    let n = points.len() as f64;
    let [u, v] = points.fold([0.0; 2], |sum, p| [sum[0] + p[0], sum[1] + p[1]]);
    [u / n, v / n]
}

/// The `--fits` CSV (see the module docs): one row per image, to a file or,
/// in tests, any writer.
#[derive(Debug)]
pub(crate) struct FitsCsv<W: Write = File> {
    out: BufWriter<W>,
}

impl FitsCsv {
    /// Create (truncate) the CSV at `path` and write its header row.
    ///
    /// # Errors
    /// Fails when the file cannot be created or the header cannot be written.
    pub(crate) fn create(path: &str) -> Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("failed to create the face fits CSV {path}"))?;
        Self::with_header(file)
    }
}

impl<W: Write> FitsCsv<W> {
    /// Start the CSV on `out` with its header row.
    ///
    /// # Errors
    /// Fails when the header cannot be written.
    fn with_header(out: W) -> Result<Self> {
        let mut out = BufWriter::new(out);
        writeln!(out, "{HEADER}").context("failed to write the face fits CSV")?;
        Ok(Self { out })
    }

    /// Append the row of image `image` (its index among the decoded 0x50e
    /// images), taken at `device_ts_us`: its fit, or `None` when the tracker
    /// found no face in it.
    ///
    /// # Errors
    /// Fails when the write fails.
    pub(crate) fn write_image(
        &mut self,
        image: u64,
        device_ts_us: u64,
        fit: Option<&FitFields<'_>>,
    ) -> Result<()> {
        self.write_row(image, device_ts_us, fit)
            .context("failed to write the face fits CSV")
    }

    /// See [`FitsCsv::write_image`].
    fn write_row(
        &mut self,
        image: u64,
        device_ts_us: u64,
        fit: Option<&FitFields<'_>>,
    ) -> std::io::Result<()> {
        let out = &mut self.out;
        write!(out, "{image},{device_ts_us},")?;
        let Some(fit) = fit else {
            write!(out, "0")?;
            for _ in 0..FIT_FIELDS {
                write!(out, ",")?;
            }
            return writeln!(out);
        };
        write!(
            out,
            "1,{},{}",
            f64::from(fit.score),
            u8::from(fit.found_by_detector)
        )?;
        for v in fit
            .rotation
            .as_flattened()
            .iter()
            .chain(&fit.translation_mm)
        {
            write!(out, ",{v}")?;
        }
        for [u, v] in image_points(fit.landmarks) {
            write!(out, ",{u},{v}")?;
        }
        writeln!(out)
    }

    /// Write out what is still buffered. Dropping the CSV would do the same
    /// but drop an error.
    ///
    /// # Errors
    /// Fails when the write fails.
    pub(crate) fn finish(mut self) -> Result<()> {
        self.out
            .flush()
            .context("failed to write the face fits CSV")
    }
}

/// The `--landmarks` file (see the module docs): one record of
/// [`LANDMARK_RECORD_BYTES`] per image, to a file or, in tests, any writer.
#[derive(Debug)]
pub(crate) struct LandmarksF32<W: Write = File> {
    out: BufWriter<W>,
}

impl LandmarksF32 {
    /// Create (truncate) the file at `path`.
    ///
    /// # Errors
    /// Fails when the file cannot be created.
    pub(crate) fn create(path: &str) -> Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("failed to create the landmarks file {path}"))?;
        Ok(Self::new(file))
    }
}

impl<W: Write> LandmarksF32<W> {
    /// Write the records to `out`.
    fn new(out: W) -> Self {
        Self {
            out: BufWriter::new(out),
        }
    }

    /// Append the record of the next image: its fit's landmarks, or `None`
    /// when the tracker found no face in it (a record of NaNs).
    ///
    /// # Errors
    /// Fails when the write fails.
    // reason: the landmarks are rounded from the fit's f64 to the f32 the
    // file holds, as its layout says (num-cast-try-from).
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn write_image(&mut self, landmarks: Option<&[[f64; 2]; LANDMARKS]>) -> Result<()> {
        let mut record = [0u8; LANDMARK_RECORD_BYTES];
        let values = record.as_chunks_mut::<4>().0;
        match landmarks {
            Some(landmarks) => {
                for (bytes, &v) in values.iter_mut().zip(landmarks.as_flattened()) {
                    *bytes = (v as f32).to_le_bytes(); // cast: the file's f32
                }
            }
            None => values.fill(f32::NAN.to_le_bytes()),
        }
        self.out
            .write_all(&record)
            .context("failed to write the landmarks file")
    }

    /// Write out what is still buffered. Dropping the writer would do the
    /// same but drop an error.
    ///
    /// # Errors
    /// Fails when the write fails.
    pub(crate) fn finish(mut self) -> Result<()> {
        self.out
            .flush()
            .context("failed to write the landmarks file")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    /// A writer the test reads back after handing a clone to a sink.
    #[derive(Clone, Default)]
    struct Shared(Rc<RefCell<Vec<u8>>>);

    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.borrow_mut().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Shared {
        fn bytes(&self) -> Vec<u8> {
            self.0.borrow().clone()
        }
    }

    /// Landmarks no two of which are alike, none a short decimal: landmark
    /// k at (k / 3 + 0.1, 280 - k / 7).
    fn landmarks() -> Box<[[f64; 2]; LANDMARKS]> {
        Box::new(std::array::from_fn(|k| {
            let k = k as f64;
            [k / 3.0 + 0.1, 280.0 - k / 7.0]
        }))
    }

    /// A fit with a rotation, translation and score that need every digit:
    /// element (i, j) of the rotation is (3i + j + 1) / 9 + 0.1.
    fn fit(landmarks: &[[f64; 2]; LANDMARKS]) -> FitFields<'_> {
        FitFields {
            rotation: std::array::from_fn(|i| {
                std::array::from_fn(|j| (3 * i + j + 1) as f64 / 9.0 + 0.1)
            }),
            translation_mm: [-12.0 / 7.0, 1e-7 / 3.0, 655.0 + 1.0 / 3.0],
            landmarks,
            score: 12.345_678,
            found_by_detector: true,
        }
    }

    /// The CSV of `rows`, each (image, device time, fit), split into fields.
    fn csv(rows: &[(u64, u64, Option<&FitFields<'_>>)]) -> Vec<Vec<String>> {
        let written = Shared::default();
        let mut csv = FitsCsv::with_header(written.clone()).expect("header");
        for &(image, ts, fit) in rows {
            csv.write_image(image, ts, fit).expect("row");
        }
        csv.finish().expect("flushed");
        let text = String::from_utf8(written.bytes()).expect("utf-8");
        text.lines()
            .map(|line| line.split(',').map(String::from).collect())
            .collect()
    }

    /// `field` read back as the f64 it was written from, bit for bit.
    fn bits(field: &str) -> u64 {
        field
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("{field:?}: {e}"))
            .to_bits()
    }

    /// The mean of the landmarks at `indices`, summed in order.
    fn mean(landmarks: &[[f64; 2]; LANDMARKS], indices: &[usize]) -> [f64; 2] {
        let mut sum = [0.0; 2];
        for &k in indices {
            sum = [sum[0] + landmarks[k][0], sum[1] + landmarks[k][1]];
        }
        [sum[0] / indices.len() as f64, sum[1] / indices.len() as f64]
    }

    #[test]
    fn the_header_names_each_of_the_29_columns_once() {
        let columns: Vec<&str> = HEADER.split(',').collect();
        assert_eq!(columns.len(), 3 + FIT_FIELDS);
        assert_eq!(columns.len(), 29);
        assert_eq!(columns.iter().collect::<HashSet<_>>().len(), 29);
        assert_eq!(
            columns,
            [
                "image_idx",
                "device_ts_us",
                "face",
                "score",
                "found_by_detector",
                "r_cam_00",
                "r_cam_01",
                "r_cam_02",
                "r_cam_10",
                "r_cam_11",
                "r_cam_12",
                "r_cam_20",
                "r_cam_21",
                "r_cam_22",
                "t_cam_x_mm",
                "t_cam_y_mm",
                "t_cam_z_mm",
                "centroid_u",
                "centroid_v",
                "nose_tip_u",
                "nose_tip_v",
                "eye_image_left_u",
                "eye_image_left_v",
                "eye_image_right_u",
                "eye_image_right_v",
                "corner_33_u",
                "corner_33_v",
                "corner_263_u",
                "corner_263_v",
            ]
        );
    }

    /// A row carries the image, then the fit: the f32 score as its f64, the
    /// rotation row by row, the translation and the six points, each value
    /// reading back as the f64 it was.
    #[test]
    fn a_fit_is_written_in_full_under_its_columns() {
        let lm = landmarks();
        let fit = fit(&lm);
        let rows = csv(&[(41, 1_234_567_890_123, Some(&fit))]);
        let [header, row] = rows.as_slice() else {
            panic!("expected a header and one row: {rows:?}");
        };
        assert_eq!(row.len(), header.len());
        let cell = |name: &str| {
            let at = header.iter().position(|c| c == name);
            row[at.unwrap_or_else(|| panic!("no column {name}"))].as_str()
        };
        assert_eq!(cell("image_idx"), "41");
        assert_eq!(cell("device_ts_us"), "1234567890123");
        assert_eq!(cell("face"), "1");
        assert_eq!(cell("found_by_detector"), "1");
        // The f32 itself: 12.345678 read as an f64 is not it.
        assert_eq!(bits(cell("score")), f64::from(fit.score).to_bits());
        assert_eq!(cell("score"), "12.345678329467773");
        for i in 0..3 {
            for j in 0..3 {
                let name = format!("r_cam_{i}{j}");
                assert_eq!(bits(cell(&name)), fit.rotation[i][j].to_bits(), "{name}");
            }
        }
        for (axis, v) in ["x", "y", "z"].iter().zip(fit.translation_mm) {
            assert_eq!(bits(cell(&format!("t_cam_{axis}_mm"))), v.to_bits());
        }
        let points = [
            ("centroid", mean(&lm, &(0..LANDMARKS).collect::<Vec<_>>())),
            ("nose_tip", lm[1]),
            ("eye_image_left", mean(&lm, &EYE_CONTOUR_IMAGE_LEFT)),
            ("eye_image_right", mean(&lm, &EYE_CONTOUR_IMAGE_RIGHT)),
            ("corner_33", lm[33]),
            ("corner_263", lm[263]),
        ];
        for (name, [u, v]) in points {
            assert_eq!(bits(cell(&format!("{name}_u"))), u.to_bits(), "{name}");
            assert_eq!(bits(cell(&format!("{name}_v"))), v.to_bits(), "{name}");
        }
        // As Python's repr() writes them: 1 / 3 + 0.1 and 280 - 263 / 7.
        assert_eq!(cell("nose_tip_u"), "0.43333333333333335");
        assert_eq!(cell("corner_263_v"), "242.42857142857144");
    }

    /// The row of an image without a face says so and leaves the fit's 26
    /// fields empty; the next row is written as usual.
    #[test]
    fn an_image_without_a_face_leaves_the_fit_fields_empty() {
        let lm = landmarks();
        let fit = FitFields {
            found_by_detector: false,
            ..fit(&lm)
        };
        let rows = csv(&[(0, 100, None), (1, 133, Some(&fit)), (2, 166, None)]);
        assert_eq!(rows.len(), 4);
        let empty = vec![String::new(); FIT_FIELDS];
        for (row, (image, ts)) in [(&rows[1], ("0", "100")), (&rows[3], ("2", "166"))] {
            assert_eq!(row.len(), 29);
            assert_eq!(row[..3], [image, ts, "0"]);
            assert_eq!(row[3..], empty[..]);
        }
        let face = &rows[2];
        assert_eq!(face.len(), 29);
        assert_eq!(face[..5], ["1", "133", "1", "12.345678329467773", "0"]);
        assert!(face[5..].iter().all(|field| !field.is_empty()));
    }

    /// The points are the mean of all 468 landmarks, landmark 1, the means
    /// of the two 16-point eye contours and landmarks 33 and 263: placed
    /// alone, each moves its own point and no other.
    #[test]
    fn each_point_is_taken_from_its_own_landmarks() {
        let base = [[140.0, 140.0]; LANDMARKS];
        assert_eq!(image_points(&base), [[140.0, 140.0]; 6]);
        let moved = |indices: &[usize]| {
            let mut lm = base;
            for &k in indices {
                lm[k] = [140.0 + 16.0, 140.0 - 32.0];
            }
            image_points(&lm)
        };
        // 468 landmarks moved by (16, -32) move their mean by as much, one
        // of them by 1/468 of that (the sums are exact); one of an eye's 16
        // moves that eye by 1/16.
        let all: Vec<usize> = (0..LANDMARKS).collect();
        assert_eq!(moved(&all)[0], [156.0, 108.0]);
        let points = moved(&[NOSE_TIP]);
        assert_eq!(points[0], [65536.0 / 468.0, 65488.0 / 468.0]);
        assert_eq!(points[1], [156.0, 108.0]);
        assert_eq!(points[2..], [[140.0, 140.0]; 4]);
        let eye_left = moved(&EYE_CONTOUR_IMAGE_LEFT[1..2]);
        assert_eq!(eye_left[2], [141.0, 138.0]);
        assert_eq!(eye_left[3..], [[140.0, 140.0]; 3]);
        let eye_right = moved(&EYE_CONTOUR_IMAGE_RIGHT[1..2]);
        assert_eq!(eye_right[2], [140.0, 140.0]);
        assert_eq!(eye_right[3], [141.0, 138.0]);
        assert_eq!(eye_right[4..], [[140.0, 140.0]; 2]);
        // A corner moves its own point and its eye's mean.
        let corner = moved(&[33]);
        assert_eq!(corner[2], [141.0, 138.0]);
        assert_eq!(corner[4], [156.0, 108.0]);
        assert_eq!(corner[5], [140.0, 140.0]);
        let corner = moved(&[263]);
        assert_eq!(corner[3], [141.0, 138.0]);
        assert_eq!(corner[4], [140.0, 140.0]);
        assert_eq!(corner[5], [156.0, 108.0]);
    }

    /// The eye contours are the head pose study's lists, sixteen distinct
    /// landmarks each, starting at the eye's outer corner.
    #[test]
    fn the_eye_contours_are_the_studys() {
        assert_eq!(
            EYE_CONTOUR_IMAGE_LEFT,
            [
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
            ]
        );
        assert_eq!(
            EYE_CONTOUR_IMAGE_RIGHT,
            [
                263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466
            ]
        );
        for (contour, corner) in [
            (EYE_CONTOUR_IMAGE_LEFT, EYE_CORNERS[0]),
            (EYE_CONTOUR_IMAGE_RIGHT, EYE_CORNERS[1]),
        ] {
            assert_eq!(contour[0], corner);
            assert_eq!(contour.iter().collect::<HashSet<_>>().len(), 16);
            assert!(contour.iter().all(|&k| k < LANDMARKS));
        }
    }

    /// Each image takes 3744 bytes: its 468 (u, v) pairs in landmark order,
    /// little-endian f32; an image without a face is all NaN.
    #[test]
    fn landmarks_are_little_endian_f32_pairs_per_image() {
        let lm = landmarks();
        let written = Shared::default();
        let mut file = LandmarksF32::new(written.clone());
        file.write_image(Some(&lm)).expect("face");
        file.write_image(None).expect("no face");
        file.write_image(Some(&lm)).expect("face again");
        file.finish().expect("flushed");
        let bytes = written.bytes();
        assert_eq!(LANDMARK_RECORD_BYTES, 3744);
        assert_eq!(bytes.len(), 3 * 3744);

        let value = |image: usize, landmark: usize, axis: usize| {
            let at = image * 3744 + (landmark * 2 + axis) * 4;
            f32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
        };
        for image in [0, 2] {
            for (k, p) in lm.iter().enumerate() {
                for (axis, &v) in p.iter().enumerate() {
                    #[allow(clippy::cast_possible_truncation)] // the file's f32
                    let want = v as f32;
                    assert_eq!(value(image, k, axis).to_bits(), want.to_bits());
                }
            }
        }
        assert!((0..LANDMARKS).all(|k| value(1, k, 0).is_nan() && value(1, k, 1).is_nan()));
        // Landmark 3 is (1.1, 279.571...): u's f32 0x3f8ccccd, low byte first.
        assert_eq!(bytes[24..28], [0xcd, 0xcc, 0x8c, 0x3f]);
    }
}
