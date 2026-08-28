//! The Metal and WGSL kernels must be the same program.
//!
//! mir ships every GPU pass twice — MSL for Apple, WGSL everywhere else — and
//! picks one at compile time. Nothing in the type system ties the two together,
//! so a change made to one and forgotten in the other compiles, runs, and shows
//! up only as a Mac and a phone drawing the same graph differently. That is
//! exactly the kind of drift these tests exist to catch.
//!
//! The check is the ordered sequence of numeric constants in each kernel. It is
//! deliberately narrow: it will not notice a rearranged expression, but it does
//! notice a threshold, a weight, a colour or a falloff changed on one side
//! only, which is what actually happens when someone edits a shader.
//!
//! When this fails, the fix is to make the edit in both languages — not to
//! relax the test. Note that an integer literal is invisible here, so write
//! floats as floats in both: `float3(0.0f, 0.0f, 1.0f)`, never `float3(0,0,1)`.

/// Every float literal in `src`, in order, with MSL's `f` suffix removed and
/// comments stripped so prose about the numbers does not count as numbers.
fn float_literals(src: &str) -> Vec<String> {
    let no_comments: String = src
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");

    let bytes: Vec<char> = no_comments.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        // A literal must start a token: `vec3` and `float4x4` are not numbers.
        if i > 0 && (bytes[i - 1].is_alphanumeric() || bytes[i - 1] == '_') {
            while i < bytes.len() && (bytes[i].is_alphanumeric() || bytes[i] == '_') {
                i += 1;
            }
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let mut is_float = false;
        if i < bytes.len() && bytes[i] == '.' {
            is_float = true;
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < bytes.len() && (bytes[i] == 'e' || bytes[i] == 'E') {
            let save = i;
            i += 1;
            if i < bytes.len() && (bytes[i] == '-' || bytes[i] == '+') {
                i += 1;
            }
            if i < bytes.len() && bytes[i].is_ascii_digit() {
                is_float = true;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            } else {
                i = save;
            }
        }
        let text: String = bytes[start..i].iter().collect();
        // MSL spells a float literal `0.5f`; WGSL spells it `0.5`.
        if i < bytes.len() && bytes[i] == 'f' && is_float {
            i += 1;
        }
        // WGSL spells an unsigned integer `2u`; MSL spells it `2`. Neither is
        // a float, and both are skipped.
        if i < bytes.len() && bytes[i] == 'u' && !is_float {
            i += 1;
        }
        if is_float {
            out.push(text);
        }
    }
    out
}

fn assert_parity(name: &str, msl: &str, wgsl: &str) {
    let a = float_literals(msl);
    let b = float_literals(wgsl);

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x, y,
            "{name}: constant #{i} differs — MSL has {x}, WGSL has {y}. \
             The two kernels have drifted; make the change in both."
        );
    }
    assert_eq!(
        a.len(),
        b.len(),
        "{name}: MSL has {} float constants, WGSL has {}. \
         Extra on one side: MSL {:?} / WGSL {:?}",
        a.len(),
        b.len(),
        &a[b.len().min(a.len())..],
        &b[a.len().min(b.len())..],
    );
}

#[test]
fn paint_kernels_agree() {
    assert_parity(
        "paint",
        mir::frame::paint::PAINT_MSL,
        mir::frame::paint::PAINT_WGSL,
    );
}

#[test]
fn cull_kernels_agree() {
    assert_parity(
        "bvh_cull",
        mir::frame::cull::BVH_CULL_MSL,
        mir::frame::cull::BVH_CULL_WGSL,
    );
}

/// Both kernels bind the same uniform, so both must declare every field of it.
/// A field added on one side only is silently read as garbage on the other.
#[test]
fn camera_uniform_declared_the_same_in_both() {
    for field in [
        "view_proj", "planes", "viewport", "near", "far",
        "cam_pos", "cam_right", "cam_up", "cam_fwd",
    ] {
        for (name, src) in [
            ("PAINT_MSL", mir::frame::paint::PAINT_MSL),
            ("PAINT_WGSL", mir::frame::paint::PAINT_WGSL),
            ("BVH_CULL_MSL", mir::frame::cull::BVH_CULL_MSL),
            ("BVH_CULL_WGSL", mir::frame::cull::BVH_CULL_WGSL),
        ] {
            assert!(
                src.contains(field),
                "{name} does not declare Camera::{field}; the struct it binds \
                 no longer matches the one the host writes."
            );
        }
    }
}

/// The light is written out normalized in both languages rather than computed,
/// so it is a literal that could drift. Nail it down.
#[test]
fn both_kernels_are_lit_by_the_same_light() {
    for (name, src) in [
        ("PAINT_MSL", mir::frame::paint::PAINT_MSL),
        ("PAINT_WGSL", mir::frame::paint::PAINT_WGSL),
    ] {
        assert!(
            src.contains("0.36370") && src.contains("0.72739") && src.contains("0.58191"),
            "{name} no longer carries the shared light direction"
        );
    }
    // And it really is a unit vector, so the diffuse term means what it says.
    let n: f32 = 0.36370f32.powi(2) + 0.72739f32.powi(2) + 0.58191f32.powi(2);
    assert!((n - 1.0).abs() < 1e-4, "light is not normalized: |L|^2 = {n}");
}
