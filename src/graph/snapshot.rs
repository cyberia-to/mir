//! mmap-backed `.graph` reader. Parses TOML frontmatter, locates the
//! `cyberlinks` binary section, iterates 128-byte fixed records zero-copy.
//!
//! Record layout (cyb-graph spec §cyberlinks):
//! ```text
//! [0..32]    ν  neuron id       (hemera hash)
//! [32..64]   p  from particle   (hemera hash)
//! [64..96]   q  to particle     (hemera hash)
//! [96..100]  τ  token index     (u32 le)
//! [100..116] a  stake amount    (u128 le)
//! [116..117] v  valence         (i8: −1/0/+1)
//! [117..125] t  block height    (u64 le)
//! [125..128] _  padding
//! ```

use std::{collections::HashMap, path::Path};

use memmap2::Mmap;
use serde::Deserialize;

use crate::error::{MirError, Result};

pub const RECORD_SIZE: usize = 128;

#[derive(Debug, Clone, Copy)]
pub struct Cyberlink {
    pub neuron: [u8; 32],
    pub from:   [u8; 32],
    pub to:     [u8; 32],
    pub token:  u32,
    pub amount: u128,
    pub valence: i8,
    pub block:  u64,
}

impl Cyberlink {
    fn decode(b: &[u8; RECORD_SIZE]) -> Self {
        let mut neuron = [0u8; 32]; neuron.copy_from_slice(&b[0..32]);
        let mut from   = [0u8; 32]; from.copy_from_slice(&b[32..64]);
        let mut to     = [0u8; 32]; to.copy_from_slice(&b[64..96]);
        Self {
            neuron,
            from,
            to,
            token:   u32::from_le_bytes(b[96..100].try_into().unwrap()),
            amount:  u128::from_le_bytes(b[100..116].try_into().unwrap()),
            valence: b[116] as i8,
            block:   u64::from_le_bytes(b[117..125].try_into().unwrap()),
        }
    }
}

pub struct CyberlinkIter<'a> {
    bytes: &'a [u8],
    pos:   usize,
}

impl<'a> Iterator for CyberlinkIter<'a> {
    type Item = Cyberlink;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + RECORD_SIZE > self.bytes.len() { return None; }
        let chunk: &[u8; RECORD_SIZE] =
            self.bytes[self.pos..self.pos + RECORD_SIZE].try_into().unwrap();
        self.pos += RECORD_SIZE;
        Some(Cyberlink::decode(chunk))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = (self.bytes.len() - self.pos) / RECORD_SIZE;
        (n, Some(n))
    }
}
impl ExactSizeIterator for CyberlinkIter<'_> {}

// ── frontmatter ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Frontmatter {
    cyb:   CybMeta,
    #[serde(default)]
    files: Vec<FileEntry>,
}

#[derive(Deserialize)]
struct CybMeta {
    types: Vec<String>,
    name:  String,
}

#[derive(Deserialize, Clone)]
struct FileEntry {
    name:   String,
    #[allow(dead_code)]
    format: String,
    #[serde(default)]
    size:   Option<u64>,
}

fn split_frontmatter(bytes: &[u8]) -> Result<(&str, usize)> {
    let mut i = 0;
    while i + 4 <= bytes.len() {
        if &bytes[i..i + 4] == b"\n~~~" {
            let s = std::str::from_utf8(&bytes[..i + 1])
                .map_err(|e| MirError::InvalidGraph(format!("non-utf8 frontmatter: {e}")))?;
            return Ok((s, i + 1));
        }
        i += 1;
    }
    Err(MirError::InvalidGraph("no `~~~` delimiter found".into()))
}

// ── GraphSnapshot ─────────────────────────────────────────────────────────────

pub struct GraphSnapshot {
    mmap:     Mmap,
    name:     String,
    sections: HashMap<String, (usize, usize)>, // name → (start, end) byte offsets
}

impl GraphSnapshot {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mmap = unsafe { Mmap::map(&std::fs::File::open(path)?) }?;

        let (fm_str, body_start) = split_frontmatter(&mmap)?;
        let fm: Frontmatter = toml::from_str(fm_str)?;

        if !fm.cyb.types.iter().any(|t| t == "graph") {
            return Err(MirError::InvalidGraph(format!(
                "types {:?} does not include \"graph\"", fm.cyb.types
            )));
        }

        let sections = index_sections(&mmap, body_start, &fm.files)?;
        Ok(Self { mmap, name: fm.cyb.name, sections })
    }

    pub fn name(&self) -> &str { &self.name }

    pub fn particle_count(&self) -> usize {
        self.section_bytes("cyberlinks")
            .map(|b| b.len() / RECORD_SIZE * 2) // upper bound; dedup in vocab
            .unwrap_or(0)
    }

    pub fn cyberlinks(&self) -> Result<CyberlinkIter<'_>> {
        Ok(CyberlinkIter { bytes: self.section_bytes("cyberlinks")?, pos: 0 })
    }

    fn section_bytes(&self, name: &str) -> Result<&[u8]> {
        let &(s, e) = self.sections.get(name)
            .ok_or(MirError::MissingSection("cyberlinks"))?;
        Ok(&self.mmap[s..e])
    }
}

fn index_sections(
    bytes: &[u8],
    body_start: usize,
    entries: &[FileEntry],
) -> Result<HashMap<String, (usize, usize)>> {
    let mut map = HashMap::new();
    let mut cur = body_start;
    for entry in entries {
        let header = format!("~~~{}\n", entry.name);
        if !bytes[cur..].starts_with(header.as_bytes()) {
            return Err(MirError::InvalidGraph(format!(
                "expected `{}` at byte {cur}", header.trim_end()
            )));
        }
        let start = cur + header.len();
        let end = match entry.size {
            Some(sz) => start + sz as usize,
            None => find_text_end(bytes, start),
        };
        if end > bytes.len() {
            return Err(MirError::InvalidGraph(format!(
                "section `{}` extends past EOF", entry.name
            )));
        }
        map.insert(entry.name.clone(), (start, end));
        cur = end;
        if cur < bytes.len() && bytes[cur] == b'\n' { cur += 1; }
    }
    Ok(map)
}

fn find_text_end(bytes: &[u8], start: usize) -> usize {
    let mut j = start;
    while j + 4 <= bytes.len() {
        if &bytes[j..j + 4] == b"\n~~~" { return j; }
        j += 1;
    }
    bytes.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cyberlink_decode_roundtrip() {
        let mut buf = [0u8; RECORD_SIZE];
        buf[96..100].copy_from_slice(&42u32.to_le_bytes());
        buf[100..116].copy_from_slice(&999u128.to_le_bytes());
        buf[116] = 1u8;
        buf[117..125].copy_from_slice(&12345u64.to_le_bytes());
        let c = Cyberlink::decode(&buf);
        assert_eq!(c.token, 42);
        assert_eq!(c.amount, 999);
        assert_eq!(c.valence, 1);
        assert_eq!(c.block, 12345);
    }
}
