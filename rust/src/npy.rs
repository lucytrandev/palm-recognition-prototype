//! Minimal NumPy `.npy` reader/writer for the two files the prototype uses:
//! `all_embeddings.npy` (`<f4`, 2-D, C order) and `image_names.npy` (`<U{n}`, 1-D).

use anyhow::{anyhow, bail, Result};
use std::path::Path;

fn header(bytes: &[u8]) -> Result<(String, usize)> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not a .npy file");
    }
    let (hlen, off) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        v => bail!("unsupported .npy version {v}"),
    };
    Ok((String::from_utf8_lossy(&bytes[off..off + hlen]).to_string(), off + hlen))
}

fn field<'a>(h: &'a str, key: &str) -> Result<&'a str> {
    let i = h.find(&format!("'{key}':")).ok_or_else(|| anyhow!("npy header lacks {key}"))? + key.len() + 3;
    Ok(h[i..].trim_start())
}

fn shape(h: &str) -> Result<Vec<usize>> {
    let s = field(h, "shape")?;
    let inner = &s[1..s.find(')').ok_or_else(|| anyhow!("bad shape"))?];
    Ok(inner.split(',').filter_map(|t| t.trim().parse().ok()).collect())
}

/// Returns (rows, cols, data) for a `<f4` array (1-D arrays become 1 row).
pub fn read_f32(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let b = std::fs::read(path)?;
    let (h, off) = header(&b)?;
    if !field(&h, "descr")?.starts_with("'<f4'") {
        bail!("{}: expected <f4", path.display());
    }
    if field(&h, "fortran_order")?.starts_with("True") {
        bail!("{}: Fortran order not supported", path.display());
    }
    let sh = shape(&h)?;
    let (r, c) = match sh.as_slice() { [n] => (1, *n), [r, c] => (*r, *c), _ => bail!("expected 1-D or 2-D") };
    let data: Vec<f32> = b[off..].chunks_exact(4).take(r * c).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect();
    if data.len() != r * c {
        bail!("{}: truncated", path.display());
    }
    Ok((r, c, data))
}

/// Reads a 1-D `<U{n}` array (UTF-32 little-endian, NUL-padded).
pub fn read_strings(path: &Path) -> Result<Vec<String>> {
    let b = std::fs::read(path)?;
    let (h, off) = header(&b)?;
    let d = field(&h, "descr")?;
    if !d.starts_with("'<U") {
        bail!("{}: expected <U strings", path.display());
    }
    let width: usize = d[3..d[1..].find('\'').unwrap() + 1].parse()?;
    let n = shape(&h)?.first().copied().unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let s: String = b[off + k * width * 4..off + (k + 1) * width * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .take_while(|&u| u != 0)
            .filter_map(char::from_u32)
            .collect();
        out.push(s);
    }
    Ok(out)
}

fn write(path: &Path, descr: &str, shape: &str, payload: &[u8]) -> Result<()> {
    let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    out.extend_from_slice(payload);
    std::fs::write(path, out)?;
    Ok(())
}

pub fn write_f32(path: &Path, rows: usize, cols: usize, data: &[f32]) -> Result<()> {
    let shape = if rows == 1 { format!("({cols},)") } else { format!("({rows}, {cols})") };
    write(path, "<f4", &shape, &data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}

pub fn write_strings(path: &Path, items: &[String]) -> Result<()> {
    let width = items.iter().map(|s| s.chars().count()).max().unwrap_or(1).max(1);
    let mut payload = Vec::with_capacity(items.len() * width * 4);
    for s in items {
        let mut n = 0;
        for c in s.chars() {
            payload.extend_from_slice(&(c as u32).to_le_bytes());
            n += 1;
        }
        payload.extend(std::iter::repeat(0u8).take((width - n) * 4));
    }
    write(path, &format!("<U{width}"), &format!("({},)", items.len()), &payload)
}
