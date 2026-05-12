//! CSR (Compressed Sparse Row) adjacency matrix.
//!
//! Built from a cyberlink iterator + particle index. Symmetric: each directed
//! edge (p→q) contributes both (p,q) and (q,p) with equal weight. Edge weight
//! = normalized stake amount (f32), clamped to [0,1].
//!
//! Used by the LOBPCG eigensolver and the heat-kernel BVH builder.

use crate::graph::{
    snapshot::Cyberlink,
    vocab::ParticleIndex,
};

/// Compressed Sparse Row matrix (symmetric, f32 values).
pub struct Csr {
    pub n:       usize,     // number of rows (= number of particles)
    pub row_ptr: Vec<u32>,  // length n+1; row_ptr[i] = first entry of row i
    pub col_idx: Vec<u32>,  // column indices
    pub values:  Vec<f32>,  // edge weights (same length as col_idx)
}

impl Csr {
    /// Construct an empty CSR for a zero-particle graph.
    pub fn empty() -> Self {
        Self { n: 0, row_ptr: vec![0], col_idx: vec![], values: vec![] }
    }

    /// Build symmetric CSR from cyberlinks. Two-pass: collect edges, then
    /// sort and deduplicate (parallel edges → weight += amount, then normalize).
    pub fn build<'a>(
        links:  impl Iterator<Item = Cyberlink>,
        vocab:  &ParticleIndex,
    ) -> Self {
        let n = vocab.len();

        // Collect (row, col, raw_amount) for both directions of each link.
        let mut edges: Vec<(u32, u32, f64)> = Vec::new();
        let mut max_amount = 1.0f64;

        for link in links {
            let Some(p) = vocab.get(&link.from) else { continue };
            let Some(q) = vocab.get(&link.to)   else { continue };
            if p == q { continue; }
            let w = link.amount as f64;
            if w > max_amount { max_amount = w; }
            edges.push((p, q, w));
            edges.push((q, p, w));
        }

        // Sort by (row, col) then accumulate parallel edges.
        edges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut col_idx: Vec<u32> = Vec::with_capacity(edges.len());
        let mut values:  Vec<f32> = Vec::with_capacity(edges.len());
        let mut row_ptr: Vec<u32> = vec![0u32; n + 1];

        let mut ei = 0;
        while ei < edges.len() {
            let (r, c, mut w) = edges[ei];
            ei += 1;
            while ei < edges.len() && edges[ei].0 == r && edges[ei].1 == c {
                w += edges[ei].2;
                ei += 1;
            }
            col_idx.push(c);
            values.push((w / max_amount) as f32);
            row_ptr[r as usize + 1] += 1;
        }

        // Convert counts to prefix sums.
        for i in 0..n {
            row_ptr[i + 1] += row_ptr[i];
        }

        Self { n, row_ptr, col_idx, values }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize { self.col_idx.len() }

    /// Row slice: (col_indices, values) for row i.
    pub fn row(&self, i: usize) -> (&[u32], &[f32]) {
        let s = self.row_ptr[i]     as usize;
        let e = self.row_ptr[i + 1] as usize;
        (&self.col_idx[s..e], &self.values[s..e])
    }

    /// Degree of particle i (number of neighbors).
    pub fn degree(&self, i: usize) -> u32 {
        self.row_ptr[i + 1] - self.row_ptr[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{snapshot::Cyberlink, vocab::ParticleIndex};

    fn hash(v: u8) -> [u8; 32] { let mut h = [0u8; 32]; h[0] = v; h }

    fn link(from: u8, to: u8, amount: u128) -> Cyberlink {
        Cyberlink { neuron: [0u8;32], from: hash(from), to: hash(to),
                    token: 0, amount, valence: 1, block: 1 }
    }

    #[test]
    fn triangle_symmetric() {
        let links = vec![link(0,1,10), link(1,2,20), link(0,2,30)];
        let vocab = ParticleIndex::build(links.iter().copied());
        let csr   = Csr::build(links.into_iter(), &vocab);

        assert_eq!(csr.n, 3);
        assert_eq!(csr.nnz(), 6); // 3 edges × 2 directions

        // Every node should have degree 2 (triangle).
        for i in 0..3 {
            assert_eq!(csr.degree(i), 2, "degree of {i}");
        }
    }

    #[test]
    fn parallel_edges_accumulate() {
        let links = vec![link(0,1,10), link(0,1,20)]; // two links same direction
        let vocab = ParticleIndex::build(links.iter().copied());
        let csr   = Csr::build(links.into_iter(), &vocab);
        // Should deduplicate to 1 undirected edge each way.
        assert_eq!(csr.nnz(), 2);
    }
}
