//! Outer contours of 8-connected foreground components (Suzuki–Abe border following, as used by
//! OpenCV `findContours(..., RETR_EXTERNAL, ...)`) and contour moments (`cv2.moments` on a contour:
//! Green's-theorem polygon moments, sign-normalised so m00 ≥ 0).

use crate::imgproc::GrayImage;

#[derive(Clone, Debug)]
pub struct Contour {
    /// Border pixels in tracing order, (x, y).
    pub points: Vec<(i64, i64)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moments {
    pub m00: f64,
    pub m10: f64,
    pub m01: f64,
}

// 8-neighbourhood, clockwise as seen on screen (image y grows downward), starting East:
// 0:E 1:SE 2:S 3:SW 4:W 5:NW 6:N 7:NE
const DX: [i64; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
const DY: [i64; 8] = [0, 1, 1, 1, 0, -1, -1, -1];

fn dir_of(dx: i64, dy: i64) -> usize {
    (0..8).find(|&k| DX[k] == dx && DY[k] == dy).expect("not a neighbour")
}

/// Outer contours of the 8-connected foreground components that are NOT inside another component's
/// hole — what OpenCV `findContours(..., RETR_EXTERNAL, ...)` returns. A component is external iff it
/// touches the outside background: the 4-connected background region reachable from beyond the image
/// border (8-connected foreground pairs with 4-connected background).
pub fn outer_contours(mask: &GrayImage) -> Vec<Contour> {
    let (w, h) = (mask.w as i64, mask.h as i64);
    let fg = |x: i64, y: i64| x >= 0 && y >= 0 && x < w && y < h && mask.data[(y * w + x) as usize] != 0;
    // outside background: flood fill (4-connectivity) from every background pixel on the border
    let mut outside = vec![false; mask.data.len()];
    let mut q: Vec<(i64, i64)> = vec![];
    for x in 0..w {
        for y in [0, h - 1] {
            if !fg(x, y) && !outside[(y * w + x) as usize] {
                outside[(y * w + x) as usize] = true;
                q.push((x, y));
            }
        }
    }
    for y in 0..h {
        for x in [0, w - 1] {
            if !fg(x, y) && !outside[(y * w + x) as usize] {
                outside[(y * w + x) as usize] = true;
                q.push((x, y));
            }
        }
    }
    while let Some((x, y)) = q.pop() {
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && nx < w && ny < h && !fg(nx, ny) && !outside[(ny * w + nx) as usize] {
                outside[(ny * w + nx) as usize] = true;
                q.push((nx, ny));
            }
        }
    }
    let touches_outside = |x: i64, y: i64| {
        x == 0 || y == 0 || x == w - 1 || y == h - 1
            || [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| {
                let (nx, ny) = (x + dx, y + dy);
                nx >= 0 && ny >= 0 && nx < w && ny < h && outside[(ny * w + nx) as usize]
            })
    };
    let mut seen = vec![false; mask.data.len()];
    let mut out = vec![];
    let mut stack = vec![];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if mask.data[i] == 0 || seen[i] {
                continue;
            }
            // First pixel of a new component in raster order: its west neighbour is background,
            // so this is an outer-border start point.
            let contour = Contour { points: follow_border(&fg, x, y) };
            // mark the whole component, noting whether it touches the outside background
            stack.push((x, y));
            seen[i] = true;
            let mut external = false;
            while let Some((cx, cy)) = stack.pop() {
                external |= touches_outside(cx, cy);
                for k in 0..8 {
                    let (nx, ny) = (cx + DX[k], cy + DY[k]);
                    if fg(nx, ny) {
                        let j = (ny * w + nx) as usize;
                        if !seen[j] {
                            seen[j] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }
            if external {
                out.push(contour);
            }
        }
    }
    out
}

/// Suzuki–Abe border following from an outer-border start (x0, y0) whose west neighbour is background.
fn follow_border(fg: &impl Fn(i64, i64) -> bool, x0: i64, y0: i64) -> Vec<(i64, i64)> {
    // 3.1: from (i2, j2) = west neighbour, search clockwise around (x0, y0) for a foreground pixel
    let start_dir = dir_of(-1, 0);
    let mut first = None;
    for s in 1..=8 {
        // clockwise on screen (y down) = increasing index in DX/DY; W itself is checked last, as OpenCV does
        let k = (start_dir + s) % 8;
        if fg(x0 + DX[k], y0 + DY[k]) {
            first = Some(k);
            break;
        }
    }
    let Some(k1) = first else { return vec![(x0, y0)] }; // isolated pixel
    let p1 = (x0 + DX[k1], y0 + DY[k1]);
    let (mut p2, mut p3) = (p1, (x0, y0));
    let mut pts = vec![];
    loop {
        // 3.3: starting from the next element after p2 counter-clockwise, find foreground p4 around p3
        let d2 = dir_of(p2.0 - p3.0, p2.1 - p3.1);
        let mut p4 = p3;
        for s in 1..=8 {
            let k = (d2 + 8 - s) % 8; // counter-clockwise on screen = decreasing index
            let q = (p3.0 + DX[k], p3.1 + DY[k]);
            if fg(q.0, q.1) {
                p4 = q;
                break;
            }
        }
        pts.push(p3);
        // 3.5: stop when back at the start with the same first move
        if p4 == (x0, y0) && p3 == p1 {
            break;
        }
        p2 = p3;
        p3 = p4;
    }
    pts
}

/// `cv2.moments(contour)` (zeroth and first order), matching OpenCV's `contourMoments`.
pub fn moments(c: &Contour) -> Moments {
    let n = c.points.len();
    if n < 2 {
        return Moments { m00: 0.0, m10: 0.0, m01: 0.0 };
    }
    let (mut a00, mut a10, mut a01) = (0f64, 0f64, 0f64);
    let (mut xp, mut yp) = (c.points[n - 1].0 as f64, c.points[n - 1].1 as f64);
    for &(x, y) in &c.points {
        let (xi, yi) = (x as f64, y as f64);
        let dxy = xp * yi - xi * yp;
        a00 += dxy;
        a10 += dxy * (xp + xi);
        a01 += dxy * (yp + yi);
        xp = xi;
        yp = yi;
    }
    let sign = if a00 < 0.0 { -1.0 } else { 1.0 };
    Moments { m00: a00 * 0.5 * sign, m10: a10 / 6.0 * sign, m01: a01 / 6.0 * sign }
}

/// `cv2.contourArea(contour)` (unsigned).
pub fn area(c: &Contour) -> f64 {
    moments(c).m00
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(w: usize, h: usize, on: &[(usize, usize)]) -> GrayImage {
        let mut m = GrayImage::new(w, h, 0);
        for &(x, y) in on {
            m.data[y * w + x] = 255;
        }
        m
    }

    #[test]
    fn filled_square_area_matches_opencv_convention() {
        // a 4×4 block of pixels: OpenCV's contour runs through pixel centres → area (4−1)² = 9
        let on: Vec<_> = (2..6).flat_map(|y| (2..6).map(move |x| (x, y))).collect();
        let cs = outer_contours(&mask(10, 10, &on));
        assert_eq!(cs.len(), 1);
        let m = moments(&cs[0]);
        assert_eq!(m.m00, 9.0);
        assert_eq!((m.m10 / m.m00, m.m01 / m.m00), (3.5, 3.5));
    }

    #[test]
    fn components_counted_and_single_pixels_have_zero_area() {
        let cs = outer_contours(&mask(10, 10, &[(1, 1), (5, 5), (6, 5), (8, 8)]));
        assert_eq!(cs.len(), 3);
        assert_eq!(area(&cs[0]), 0.0);
    }
}
