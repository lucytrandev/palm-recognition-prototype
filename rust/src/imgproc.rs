//! Grayscale primitives that follow OpenCV bit-for-bit (ported from dhaneshkk/test1_palm):
//! CLAHE, INTER_LINEAR resize (SIMD fixed-point path), BGR/RGB→gray, plus the
//! `COLOR_RGB2GRAY`-on-BGR quirk the Python prototype has (see `RgbImage::to_gray_prototype`).

#[derive(Clone, Debug)]
pub struct GrayImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

/// Interleaved 8-bit RGB frame (teachers run on colour; the runtime path is grayscale).
#[derive(Clone, Debug)]
pub struct RgbImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn from_dynamic(img: &image::DynamicImage) -> Self {
        let rgb = img.to_rgb8();
        Self { w: rgb.width() as usize, h: rgb.height() as usize, data: rgb.into_raw() }
    }
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        Ok(Self::from_dynamic(&image::open(path)?))
    }
    /// OpenCV `cvtColor(COLOR_BGR2GRAY)` fixed point: (9798·R + 19235·G + 3735·B + 2¹⁴) >> 15
    /// (verified identical over all 2²⁴ colours against OpenCV 4.x).
    pub fn to_gray(&self) -> GrayImage {
        let data = self
            .data
            .chunks_exact(3)
            .map(|p| ((9798 * p[0] as u32 + 19235 * p[1] as u32 + 3735 * p[2] as u32 + (1 << 14)) >> 15) as u8)
            .collect();
        GrayImage { w: self.w, h: self.h, data }
    }
    /// What the Python prototype computes: `cv2.imread` gives BGR, then `cvtColor(..., COLOR_RGB2GRAY)`
    /// treats B as R. Kept on purpose so embeddings match the committed database.
    pub fn to_gray_prototype(&self) -> GrayImage {
        let data = self
            .data
            .chunks_exact(3)
            .map(|p| ((9798 * p[2] as u32 + 19235 * p[1] as u32 + 3735 * p[0] as u32 + (1 << 14)) >> 15) as u8)
            .collect();
        GrayImage { w: self.w, h: self.h, data }
    }
    /// Split into three planes (R, G, B).
    pub fn planes(&self) -> [GrayImage; 3] {
        let mut out = [GrayImage::new(self.w, self.h, 0), GrayImage::new(self.w, self.h, 0), GrayImage::new(self.w, self.h, 0)];
        for (i, p) in self.data.chunks_exact(3).enumerate() {
            for c in 0..3 {
                out[c].data[i] = p[c];
            }
        }
        out
    }
    /// `cv2.resize(INTER_LINEAR)` applied per channel (bit-exact path).
    pub fn resize_linear(&self, dw: usize, dh: usize) -> RgbImage {
        let planes = self.planes().map(|p| resize_linear(&p, dw, dh));
        let mut data = vec![0u8; dw * dh * 3];
        for i in 0..dw * dh {
            for c in 0..3 {
                data[i * 3 + c] = planes[c].data[i];
            }
        }
        RgbImage { w: dw, h: dh, data }
    }
}

#[derive(Clone, Debug)]
pub struct FloatImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

#[inline]
fn reflect101(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let mut i = i;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * n - 2 - i;
        } else {
            return i as usize;
        }
    }
}

/// OpenCV `saturate_cast<uchar>(float)`: round-half-to-even, then clamp.
#[inline]
pub fn sat_u8(v: f32) -> u8 {
    v.round_ties_even().clamp(0.0, 255.0) as u8
}

impl GrayImage {
    pub fn new(w: usize, h: usize, fill: u8) -> Self {
        Self { w, h, data: vec![fill; w * h] }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.w + x]
    }

    pub fn from_dynamic(img: &image::DynamicImage) -> Self {
        RgbImage::from_dynamic(img).to_gray()
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        Ok(Self::from_dynamic(&image::open(path)?))
    }

    pub fn crop(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Self {
        let (w, h) = (x1 - x0, y1 - y0);
        let mut data = Vec::with_capacity(w * h);
        for y in y0..y1 {
            data.extend_from_slice(&self.data[y * self.w + x0..y * self.w + x1]);
        }
        Self { w, h, data }
    }

    pub fn to_jpeg(&self, quality: u8) -> Vec<u8> {
        let mut out = Vec::new();
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
        enc.encode(&self.data, self.w as u32, self.h as u32, image::ExtendedColorType::L8)
            .expect("jpeg encode");
        out
    }

    /// Draw a hollow rectangle (thickness `t`) with grey level `v` — used for the live MJPEG view.
    pub fn draw_rect(&mut self, b: [f32; 4], v: u8, t: usize) {
        let x0 = (b[0].max(0.0) as usize).min(self.w.saturating_sub(1));
        let y0 = (b[1].max(0.0) as usize).min(self.h.saturating_sub(1));
        let x1 = (b[2].max(0.0) as usize).min(self.w.saturating_sub(1));
        let y1 = (b[3].max(0.0) as usize).min(self.h.saturating_sub(1));
        for k in 0..t {
            for x in x0..=x1 {
                for y in [y0 + k, y1.saturating_sub(k)] {
                    if y < self.h {
                        self.data[y * self.w + x] = v;
                    }
                }
            }
            for y in y0..=y1 {
                for x in [x0 + k, x1.saturating_sub(k)] {
                    if x < self.w {
                        self.data[y * self.w + x] = v;
                    }
                }
            }
        }
    }
}

/// OpenCV CLAHE (8-bit, histSize 256).
pub fn clahe(src: &GrayImage, clip_limit: f64, tiles_x: usize, tiles_y: usize) -> GrayImage {
    let (w, h) = (src.w, src.h);
    // OpenCV pads by (tiles - rem) in BOTH dims whenever either is non-divisible.
    let (ew, eh) = if w % tiles_x == 0 && h % tiles_y == 0 {
        (w, h)
    } else {
        (w + tiles_x - w % tiles_x, h + tiles_y - h % tiles_y)
    };
    let (tw, th) = (ew / tiles_x, eh / tiles_y);
    let tile_total = tw * th;
    let lut_scale = 255.0f32 / tile_total as f32;
    let clip = ((clip_limit * tile_total as f64 / 256.0) as i64).max(1);
    let mut luts = vec![[0u8; 256]; tiles_x * tiles_y];
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let mut hist = [0i64; 256];
            for y in ty * th..(ty + 1) * th {
                let sy = reflect101(y as isize, h);
                for x in tx * tw..(tx + 1) * tw {
                    let sx = reflect101(x as isize, w);
                    hist[src.data[sy * w + sx] as usize] += 1;
                }
            }
            let mut clipped = 0i64;
            for v in hist.iter_mut() {
                if *v > clip {
                    clipped += *v - clip;
                    *v = clip;
                }
            }
            let batch = clipped / 256;
            let mut residual = clipped - batch * 256;
            for v in hist.iter_mut() {
                *v += batch;
            }
            if residual != 0 {
                let step = ((256 / residual) as usize).max(1);
                let mut i = 0usize;
                while i < 256 && residual > 0 {
                    hist[i] += 1;
                    i += step;
                    residual -= 1;
                }
            }
            let lut = &mut luts[ty * tiles_x + tx];
            let mut sum = 0i64;
            for i in 0..256 {
                sum += hist[i];
                lut[i] = sat_u8(sum as f32 * lut_scale);
            }
        }
    }
    let inv_tw = 1.0f32 / tw as f32;
    let inv_th = 1.0f32 / th as f32;
    let xs: Vec<(usize, usize, f32)> = (0..w)
        .map(|x| {
            let txf = x as f32 * inv_tw - 0.5;
            let tx1 = txf.floor() as isize;
            let xa = txf - tx1 as f32;
            let t1 = tx1.max(0) as usize;
            let t2 = ((tx1 + 1) as usize).min(tiles_x - 1);
            (t1, t2, xa)
        })
        .collect();
    let mut dst = GrayImage::new(w, h, 0);
    for y in 0..h {
        let tyf = y as f32 * inv_th - 0.5;
        let ty1 = tyf.floor() as isize;
        let ya = tyf - ty1 as f32;
        let r1 = ty1.max(0) as usize;
        let r2 = ((ty1 + 1) as usize).min(tiles_y - 1);
        for x in 0..w {
            let v = src.data[y * w + x] as usize;
            let (t1, t2, xa) = xs[x];
            let l = |r: usize, t: usize| luts[r * tiles_x + t][v] as f32;
            let res = (l(r1, t1) * (1.0 - xa) + l(r1, t2) * xa) * (1.0 - ya)
                + (l(r2, t1) * (1.0 - xa) + l(r2, t2) * xa) * ya;
            dst.data[y * w + x] = sat_u8(res);
        }
    }
    dst
}

/// OpenCV 8-bit two-tap resampling, bit-exact with its SIMD path:
/// horizontal `S0·a0 + S1·a1` (coefficients ×2048, round-half-even), vertical
/// `((mulhi(H0>>4, b0) + mulhi(H1>>4, b1)) + 2) >> 2` with 16-bit saturation (`VResizeLinearVec_32s8u`).
fn resize_fixed(src: &GrayImage, xm: &[(usize, usize, f32)], ym: &[(usize, usize, f32)]) -> GrayImage {
    let coef = |f: f32| -> (i32, i32) { (((1.0 - f) * 2048.0).round_ties_even() as i32, (f * 2048.0).round_ties_even() as i32) };
    let xc: Vec<(usize, usize, i32, i32)> = xm.iter().map(|&(a, b, f)| { let (c0, c1) = coef(f); (a, b, c0, c1) }).collect();
    let (dw, dh) = (xm.len(), ym.len());
    let hrow = |y: usize| -> Vec<i32> {
        let row = &src.data[y * src.w..(y + 1) * src.w];
        xc.iter().map(|&(a, b, c0, c1)| row[a] as i32 * c0 + row[b] as i32 * c1).collect()
    };
    let sat16 = |v: i32| v.clamp(-32768, 32767);
    let mut dst = GrayImage::new(dw, dh, 0);
    for (y, &(y0, y1, f)) in ym.iter().enumerate() {
        let (b0, b1) = coef(f);
        let (r0, r1) = (hrow(y0), hrow(y1));
        for x in 0..dw {
            let m0 = (sat16(r0[x] >> 4) * b0) >> 16;
            let m1 = (sat16(r1[x] >> 4) * b1) >> 16;
            let s = sat16(m0 + m1);
            dst.data[y * dw + x] = ((s + 2) >> 2).clamp(0, 255) as u8;
        }
    }
    dst
}

/// `cv2.resize(..., INTER_LINEAR)` for 8-bit images (half-pixel centres, edge clamping), bit-exact.
pub fn resize_linear(src: &GrayImage, dw: usize, dh: usize) -> GrayImage {
    if src.w == dw * 2 && src.h == dh * 2 {
        // OpenCV switches INTER_LINEAR with an exact 2× shrink to the fast area path: (a+b+c+d+2) >> 2
        let mut dst = GrayImage::new(dw, dh, 0);
        for y in 0..dh {
            for x in 0..dw {
                let p = |xx: usize, yy: usize| src.data[yy * src.w + xx] as u32;
                let s = p(2 * x, 2 * y) + p(2 * x + 1, 2 * y) + p(2 * x, 2 * y + 1) + p(2 * x + 1, 2 * y + 1);
                dst.data[y * dw + x] = ((s + 2) >> 2) as u8;
            }
        }
        return dst;
    }
    let map = |d: usize, n: usize, dn: usize| -> (usize, usize, f32) {
        let scale = n as f64 / dn as f64;
        let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
        let mut s = f.floor() as isize;
        let mut a = f - s as f32;
        if s < 0 {
            s = 0;
            a = 0.0;
        }
        if s as usize >= n - 1 {
            return (n - 1, n - 1, 0.0);
        }
        (s as usize, s as usize + 1, a)
    };
    let xm: Vec<_> = (0..dw).map(|x| map(x, src.w, dw)).collect();
    let ym: Vec<_> = (0..dh).map(|y| map(y, src.h, dh)).collect();
    resize_fixed(src, &xm, &ym)
}
