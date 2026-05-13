//! Particle vocabulary: hemera hash → ParticleIdx (u32).
//!
//! Insertion order follows block height (ascending). Ties broken by
//! lexicographic order of the hemera hash — deterministic across all neurons
//! on the same graph state (R-1.0 §4.1 anchor invariant).

use std::collections::HashMap;

use crate::graph::snapshot::Cyberlink;

pub type ParticleIdx = u32;

pub struct ParticleIndex {
    /// hash → idx; populated in block-height insertion order
    map:    HashMap<[u8; 32], ParticleIdx>,
    /// idx → hash; inverse of map
    hashes: Vec<[u8; 32]>,
}

impl ParticleIndex {
    /// Construct an empty ParticleIndex with no particles.
    pub fn empty() -> Self {
        Self { map: HashMap::new(), hashes: Vec::new() }
    }

    /// Build from a cyberlink iterator. Scans all `from` and `to` fields,
    /// assigns indices by first-seen block height.
    pub fn build<'a>(links: impl Iterator<Item = Cyberlink>) -> Self {
        // Collect first-seen block height for each particle hash.
        let mut first_seen: HashMap<[u8; 32], u64> = HashMap::new();
        for link in links {
            first_seen.entry(link.from).or_insert(link.block);
            first_seen.entry(link.to).or_insert(link.block);
        }

        // Sort by (block, hash) for deterministic insertion-order (§4.1).
        let mut ordered: Vec<([u8; 32], u64)> = first_seen.into_iter().collect();
        ordered.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let mut map = HashMap::with_capacity(ordered.len());
        let mut hashes = Vec::with_capacity(ordered.len());
        for (hash, _) in ordered {
            let idx = hashes.len() as ParticleIdx;
            map.insert(hash, idx);
            hashes.push(hash);
        }
        Self { map, hashes }
    }

    pub fn len(&self) -> usize { self.hashes.len() }
    pub fn is_empty(&self) -> bool { self.hashes.is_empty() }

    pub fn get(&self, hash: &[u8; 32]) -> Option<ParticleIdx> {
        self.map.get(hash).copied()
    }

    pub fn hash(&self, idx: ParticleIdx) -> &[u8; 32] {
        &self.hashes[idx as usize]
    }

    /// The anchor set: first 1024 particles by insertion order (R-1.0 §4.1).
    pub fn anchor(&self) -> &[[u8; 32]] {
        let n = self.hashes.len().min(1024);
        &self.hashes[..n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::snapshot::Cyberlink;

    fn make_link(from: u8, to: u8, block: u64) -> Cyberlink {
        let mut f = [0u8; 32]; f[0] = from;
        let mut t = [0u8; 32]; t[0] = to;
        Cyberlink { neuron: [0u8; 32], from: f, to: t, token: 0, amount: 0, valence: 0, block }
    }

    #[test]
    fn insertion_order() {
        let links = vec![
            make_link(3, 4, 10),
            make_link(1, 2, 5),
            make_link(2, 3, 7),
        ];
        let idx = ParticleIndex::build(links.into_iter());
        // particle 1 first seen at block 5 → idx 0
        let mut h1 = [0u8; 32]; h1[0] = 1;
        assert_eq!(idx.get(&h1), Some(0));
        // particle 4 first seen at block 10 → last (idx 3, 0-based)
        let mut h4 = [0u8; 32]; h4[0] = 4;
        assert_eq!(idx.get(&h4), Some(3));
        assert_eq!(idx.len(), 4);
    }
}
