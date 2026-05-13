//! §5.2 Metadata overlays: schemaless-default, crystal-v5, custom.

/// Rendering preset for metadata overlay (§5.2).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum OverlayPreset {
    /// Pure topology — §5.1 schemaless defaults only.
    #[default]
    None,
    /// crystal-v5: 6 types → shapes, 21 domains → triad palette, 5 sizes → radius.
    CrystalV5,
    /// Consumer-supplied lookup tables.
    Custom,
}

/// §5.2 particle shape classifications.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParticleShape {
    Sphere,
    Torus,
    Tetrahedron,
    Cube,
    Octahedron,
    Cylinder,
}

/// Metadata attached to a single particle (consumer-supplied, optional).
#[derive(Clone, Debug, Default)]
pub struct ParticleMetadata {
    /// crystal-type (0–5) for crystal-v5 preset.
    pub crystal_type:   Option<u8>,
    /// crystal-domain (0–20) for crystal-v5 preset.
    pub crystal_domain: Option<u8>,
    /// crystal-size (0–4) for crystal-v5 preset.
    pub crystal_size:   Option<u8>,
}

/// Output of applying an overlay to one particle.
#[derive(Clone, Debug)]
pub struct OverlayResult {
    pub color:  [f32; 3],
    pub radius: f32,
    pub shape:  ParticleShape,
}

/// Apply §5.2 overlay to produce final color/radius/shape.
///
/// `preset`      — which overlay to apply.
/// `base_color`  — schemaless color from §5.1.
/// `base_radius` — schemaless radius from §5.1.
/// `meta`        — optional particle metadata.
pub fn apply_overlay(
    preset:      &OverlayPreset,
    base_color:  [f32; 3],
    base_radius: f32,
    meta:        Option<&ParticleMetadata>,
) -> OverlayResult {
    match preset {
        OverlayPreset::None | OverlayPreset::Custom => {
            OverlayResult { color: base_color, radius: base_radius, shape: ParticleShape::Sphere }
        }
        OverlayPreset::CrystalV5 => {
            apply_crystal_v5(base_color, base_radius, meta)
        }
    }
}

fn apply_crystal_v5(
    base_color:  [f32; 3],
    base_radius: f32,
    meta:        Option<&ParticleMetadata>,
) -> OverlayResult {
    let default_meta = ParticleMetadata { crystal_type: None, crystal_domain: None, crystal_size: None };
    let m = meta.unwrap_or(&default_meta);

    // 6 crystal types → 6 shapes.
    let shape = match m.crystal_type.unwrap_or(0) {
        0 => ParticleShape::Sphere,
        1 => ParticleShape::Torus,
        2 => ParticleShape::Tetrahedron,
        3 => ParticleShape::Cube,
        4 => ParticleShape::Octahedron,
        _ => ParticleShape::Cylinder,
    };

    // 21 domains → 7 triads × 3 shades.
    let domain = m.crystal_domain.unwrap_or(0) as usize;
    let triad  = domain / 3;  // 0–6
    let shade  = domain % 3;  // 0–2

    // 7 triads: FORM / MASS / SPACE / LIFE / WORD / WORK / PLAY.
    let triad_hues: [f32; 7] = [0.0, 30.0, 120.0, 150.0, 210.0, 270.0, 330.0];
    let hue = triad_hues[triad.min(6)];
    let lum = 0.4 + 0.15 * shade as f32;  // 0.40 / 0.55 / 0.70
    let color = hsl_to_rgb(hue / 360.0 * std::f32::consts::TAU, 0.75, lum);

    // 5 sizes → radius multiplier.
    let size_mult = match m.crystal_size.unwrap_or(2) {
        0 => 0.5,
        1 => 0.75,
        2 => 1.0,
        3 => 1.4,
        _ => 2.0,
    };

    let _ = base_color; // crystal-v5 overrides color from domain
    OverlayResult { color, radius: base_radius * size_mult, shape }
}

fn hsl_to_rgb(h_rad: f32, s: f32, l: f32) -> [f32; 3] {
    let h = h_rad.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU * 360.0;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = if h < 60.0      { (c, x, 0.0) }
                    else if h < 120.0 { (x, c, 0.0) }
                    else if h < 180.0 { (0.0, c, x) }
                    else if h < 240.0 { (0.0, x, c) }
                    else if h < 300.0 { (x, 0.0, c) }
                    else              { (c, 0.0, x) };
    [(r+m).clamp(0.0,1.0), (g+m).clamp(0.0,1.0), (b+m).clamp(0.0,1.0)]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_preset_is_identity() {
        let color  = [0.5, 0.3, 0.8];
        let radius = 7.5;
        let out = apply_overlay(&OverlayPreset::None, color, radius, None);
        assert_eq!(out.color, color);
        assert_eq!(out.radius, radius);
        assert_eq!(out.shape, ParticleShape::Sphere);
    }

    #[test]
    fn crystal_type_selects_shape() {
        let meta = ParticleMetadata { crystal_type: Some(1), crystal_domain: None, crystal_size: None };
        let out = apply_overlay(&OverlayPreset::CrystalV5, [1.0, 1.0, 1.0], 1.0, Some(&meta));
        assert_eq!(out.shape, ParticleShape::Torus);
    }

    #[test]
    fn crystal_size_scales_radius() {
        let meta_small = ParticleMetadata { crystal_type: None, crystal_domain: None, crystal_size: Some(0) };
        let meta_large = ParticleMetadata { crystal_type: None, crystal_domain: None, crystal_size: Some(4) };
        let small = apply_overlay(&OverlayPreset::CrystalV5, [1.0,1.0,1.0], 1.0, Some(&meta_small));
        let large = apply_overlay(&OverlayPreset::CrystalV5, [1.0,1.0,1.0], 1.0, Some(&meta_large));
        assert!(large.radius > small.radius, "large={} small={}", large.radius, small.radius);
    }

    #[test]
    fn crystal_domain_produces_valid_color() {
        for domain in 0u8..21 {
            let meta = ParticleMetadata { crystal_type: None, crystal_domain: Some(domain), crystal_size: None };
            let out = apply_overlay(&OverlayPreset::CrystalV5, [0.5,0.5,0.5], 1.0, Some(&meta));
            for &c in &out.color {
                assert!(c >= 0.0 && c <= 1.0, "domain {} color out of range: {:?}", domain, out.color);
            }
        }
    }
}
