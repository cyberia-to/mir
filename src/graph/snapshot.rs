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
            Some(sz) => start.checked_add(sz as usize).ok_or_else(|| {
                MirError::InvalidGraph(format!("section `{}` size overflows", entry.name))
            })?,
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

    #[test]
    fn index_sections_rejects_overflowing_size() {
        let bytes = b"~~~foo\nhello".to_vec();
        let entries = vec![FileEntry {
            name: "foo".to_string(),
            format: "text".to_string(),
            size: Some(u64::MAX - 5),
        }];
        let err = index_sections(&bytes, 0, &entries).unwrap_err();
        assert!(matches!(err, MirError::InvalidGraph(_)));
    }

    #[test]
    fn index_sections_bounds_a_sized_section() {
        let bytes = b"~~~foo\nhello~~~bar\nworld".to_vec();
        let entries = vec![
            FileEntry { name: "foo".to_string(), format: "text".to_string(), size: Some(5) },
            FileEntry { name: "bar".to_string(), format: "text".to_string(), size: None },
        ];
        let sections = index_sections(&bytes, 0, &entries).unwrap();
        assert_eq!(&bytes[sections["foo"].0..sections["foo"].1], b"hello");
        assert_eq!(&bytes[sections["bar"].0..sections["bar"].1], b"world");
    }

    fn record(neuron: u8, from: u8, to: u8, token: u32, amount: u128, valence: i8, block: u64) -> [u8; RECORD_SIZE] {
        let mut buf = [0u8; RECORD_SIZE];
        buf[0..32].fill(neuron);
        buf[32..64].fill(from);
        buf[64..96].fill(to);
        buf[96..100].copy_from_slice(&token.to_le_bytes());
        buf[100..116].copy_from_slice(&amount.to_le_bytes());
        buf[116] = valence as u8;
        buf[117..125].copy_from_slice(&block.to_le_bytes());
        buf
    }

    #[test]
    fn decode_reads_every_field_at_its_documented_offset() {
        let buf = record(0x11, 0x22, 0x33, 7, 8, -1, 9);
        let c = Cyberlink::decode(&buf);
        assert_eq!(c.neuron, [0x11u8; 32]);
        assert_eq!(c.from, [0x22u8; 32]);
        assert_eq!(c.to, [0x33u8; 32]);
        assert_eq!(c.token, 7);
        assert_eq!(c.amount, 8);
        assert_eq!(c.valence, -1);
        assert_eq!(c.block, 9);
    }

    #[test]
    fn decode_valence_round_trips_full_signed_range() {
        for v in [i8::MIN, -1, 0, 1, i8::MAX] {
            let buf = record(0, 0, 0, 0, 0, v, 0);
            assert_eq!(Cyberlink::decode(&buf).valence, v);
        }
    }

    #[test]
    fn decode_never_reads_the_padding_bytes() {
        let mut a = record(1, 2, 3, 4, 5, 6, 7);
        let b = a;
        a[125..128].copy_from_slice(&[0xaa, 0xbb, 0xcc]);
        let mut c = b;
        c[125..128].copy_from_slice(&[0x00, 0x11, 0x22]);
        assert_eq!(Cyberlink::decode(&a).token, Cyberlink::decode(&c).token);
        assert_eq!(Cyberlink::decode(&a).amount, Cyberlink::decode(&c).amount);
        assert_eq!(Cyberlink::decode(&a).valence, Cyberlink::decode(&c).valence);
        assert_eq!(Cyberlink::decode(&a).block, Cyberlink::decode(&c).block);
    }

    #[test]
    fn iter_over_empty_input_yields_nothing() {
        let mut it = CyberlinkIter { bytes: &[], pos: 0 };
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none());
    }

    #[test]
    fn iter_drops_a_short_trailing_chunk_instead_of_panicking() {
        let mut bytes = record(1, 1, 1, 1, 1, 1, 1).to_vec();
        bytes.extend_from_slice(&[0u8; 10]); // short trailing chunk, not a full record
        let mut it = CyberlinkIter { bytes: &bytes, pos: 0 };
        assert!(it.next().is_some());
        assert!(it.next().is_none());
    }

    #[test]
    fn iter_yields_every_exact_record_in_order_and_tracks_len() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&record(1, 1, 1, 10, 0, 0, 0));
        bytes.extend_from_slice(&record(2, 2, 2, 20, 0, 0, 0));
        bytes.extend_from_slice(&record(3, 3, 3, 30, 0, 0, 0));
        let mut it = CyberlinkIter { bytes: &bytes, pos: 0 };
        assert_eq!(it.len(), 3);
        assert_eq!(it.next().unwrap().token, 10);
        assert_eq!(it.len(), 2);
        assert_eq!(it.next().unwrap().token, 20);
        assert_eq!(it.len(), 1);
        assert_eq!(it.next().unwrap().token, 30);
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none());
    }
}
