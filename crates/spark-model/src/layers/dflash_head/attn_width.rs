// SPDX-License-Identifier: AGPL-3.0-only
//! Drafter head-width → γ-block paged-indirect attention module (A93).
//!
//! `inferspark_prefill_paged_indirect` is compiled with a compile-time `HDIM`
//! (tile width). The DFlash drafter passes its *runtime* `head_dim` into that
//! kernel, so the two must agree: an HDIM=256 build fed a head_dim=128 drafter
//! loads heads `h` and `h+1` into one tile and every head's scores become
//! `Q_h·K_h + Q_{h+1}·K_{h+1}` (proven on GB10, spark-bench HANDOFF-08,
//! 2026-09-07; acceptance 4.55/7 → 6.17/7 once the width matched). This module
//! is the single place that maps a drafter width to a kernel and refuses any
//! width that has no specialization, so the mismatch cannot recur silently.

use anyhow::{Result, bail};

/// One compiled width of the paged-indirect γ-block attention kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedIndirectSpec {
    /// Kernel module name (`kernels/gb10/common/KERNEL.toml` `[modules]` alias).
    pub module: &'static str,
    /// `extern "C"` entry symbol.
    pub func: &'static str,
    /// Compile-time `HDIM` the module was built with.
    pub hdim: u32,
}

/// HDIM=256 build — the common default (`inferspark_prefill_paged_indirect.cu`).
pub const PAGED_INDIRECT_H256: PagedIndirectSpec = PagedIndirectSpec {
    module: "prefill_paged_indirect",
    func: "inferspark_prefill_paged_indirect",
    hdim: 256,
};

/// HDIM=128 build (`inferspark_prefill_paged_indirect_h128.cu`).
pub const PAGED_INDIRECT_H128: PagedIndirectSpec = PagedIndirectSpec {
    module: "prefill_paged_indirect_h128",
    func: "inferspark_prefill_paged_indirect_h128",
    hdim: 128,
};

/// Select the specialization whose compiled width equals the drafter's
/// `head_dim`. Any other width is an error: there is no safe fallback.
pub fn paged_indirect_spec_for(head_dim: usize) -> Result<PagedIndirectSpec> {
    match head_dim {
        128 => Ok(PAGED_INDIRECT_H128),
        256 => Ok(PAGED_INDIRECT_H256),
        other => bail!(
            "DFlash drafter head_dim={other} has no paged-indirect attention \
             specialization (compiled widths: 128, 256). Refusing to run the \
             γ-block attention with a mismatched tile width — add an \
             `inferspark_prefill_paged_indirect_h{other}` module and register it \
             in dflash_head::attn_width."
        ),
    }
}

/// Load-time guard: the module we are about to bind must have been compiled
/// for exactly the drafter's width. Kept separate from the selector so a
/// future caller that obtains a spec another way still cannot bind a
/// mismatched kernel.
pub fn assert_width(spec: PagedIndirectSpec, head_dim: usize) -> Result<()> {
    if spec.hdim as usize != head_dim {
        bail!(
            "DFlash paged-indirect attention width mismatch: module `{}` is \
             compiled HDIM={} but the drafter head_dim={head_dim}. Refusing to \
             arm the drafter (A93: this mismatch corrupts every head's softmax \
             while staying in-bounds).",
            spec.func,
            spec.hdim
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_dim_128_resolves_the_h128_specialization() {
        let s = paged_indirect_spec_for(128).unwrap();
        assert_eq!(s, PAGED_INDIRECT_H128);
        assert_eq!(s.module, "prefill_paged_indirect_h128");
        assert_eq!(s.func, "inferspark_prefill_paged_indirect_h128");
        assert_eq!(s.hdim, 128);
    }

    #[test]
    fn head_dim_256_resolves_the_existing_default_module() {
        let s = paged_indirect_spec_for(256).unwrap();
        assert_eq!(s, PAGED_INDIRECT_H256);
        assert_eq!(s.module, "prefill_paged_indirect");
        assert_eq!(s.func, "inferspark_prefill_paged_indirect");
        assert_eq!(s.hdim, 256);
    }

    #[test]
    fn unsupported_widths_are_explicit_errors() {
        for w in [0usize, 64, 96, 192, 512] {
            let e = paged_indirect_spec_for(w).unwrap_err().to_string();
            assert!(e.contains(&format!("head_dim={w}")), "{e}");
            assert!(
                e.contains("no paged-indirect attention specialization"),
                "{e}"
            );
        }
    }

    #[test]
    fn width_guard_rejects_a_mismatched_module() {
        // The exact A91 runtime: HDIM=256 module, head_dim=128 drafter.
        let e = assert_width(PAGED_INDIRECT_H256, 128)
            .unwrap_err()
            .to_string();
        assert!(e.contains("compiled HDIM=256"), "{e}");
        assert!(e.contains("head_dim=128"), "{e}");
        assert!(assert_width(PAGED_INDIRECT_H128, 256).is_err());
        assert!(assert_width(PAGED_INDIRECT_H128, 128).is_ok());
        assert!(assert_width(PAGED_INDIRECT_H256, 256).is_ok());
    }

    #[test]
    fn every_supported_width_passes_its_own_guard() {
        for w in [128usize, 256] {
            let s = paged_indirect_spec_for(w).unwrap();
            assert_width(s, w).unwrap();
        }
    }

    /// Source contract: the Rust names must match the kernel sources and the
    /// KERNEL.toml alias, and the h128 file must pin HDIM=128 under its own
    /// symbol. A drift here would resolve at runtime to the wrong width or
    /// fail the kernel lookup at boot instead of at `cargo test`.
    #[test]
    fn kernel_sources_match_the_specs() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels/gb10/common/");
        let h128 =
            std::fs::read_to_string(format!("{root}inferspark_prefill_paged_indirect_h128.cu"))
                .expect("h128 specialization source");
        assert!(
            h128.contains("#define HDIM 128"),
            "h128 file must pin HDIM 128"
        );
        assert!(
            h128.contains(&format!("#define KERNEL_NAME {}", PAGED_INDIRECT_H128.func)),
            "h128 file must define its own KERNEL_NAME"
        );
        assert!(h128.contains("#include \"inferspark_prefill_paged_indirect.cu\""));
        let common = std::fs::read_to_string(format!("{root}inferspark_prefill_paged_indirect.cu"))
            .expect("common source");
        assert!(
            common.contains(
                "#ifndef KERNEL_NAME\n#define KERNEL_NAME inferspark_prefill_paged_indirect\n#endif"
            ),
            "common file must keep KERNEL_NAME overridable"
        );
        assert!(
            !common.contains("#define HDIM"),
            "common file must not pin HDIM (256 comes from the .cuh default)"
        );
        let toml = std::fs::read_to_string(format!("{root}KERNEL.toml")).expect("KERNEL.toml");
        for s in [PAGED_INDIRECT_H128, PAGED_INDIRECT_H256] {
            let line = format!("{} = \"{}\"", s.func, s.module);
            assert!(toml.contains(&line), "KERNEL.toml must alias `{line}`");
        }
    }
}
