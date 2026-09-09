// SPDX-License-Identifier: AGPL-3.0-only

//! Drafter HF `config.json` schema: [`DflashConfig`] plus the nested
//! [`DflashRopeScaling`] / [`DflashSubConfig`] blocks and their serde
//! defaults. Split out of `dflash_loader.rs` to stay under the 500-LoC
//! cap; field semantics are documented on the types themselves.

use serde::Deserialize;

/// Drafter HF `config.json` (subset Atlas consumes). Mirrors
/// `z-lab/Qwen3.6-35B-A3B-DFlash/config.json` field names verbatim so
/// `serde_json::from_str` works directly on the raw file.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub draft_vocab_size: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Block size γ. Qwen3.6-DFlash ships `block_size: 16`.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// DFlash-specific nested config object.
    #[serde(default)]
    pub dflash_config: Option<DflashSubConfig>,
    /// Drafter base RoPE θ, TOP-LEVEL form. Defaults to 10M (matches
    /// Qwen3.6-DFlash). Read through [`DflashConfig::effective_rope_theta`],
    /// which lets a nested `rope_parameters.rope_theta` win.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// RMSNorm epsilon for every drafter norm (`hidden_norm`, per-layer
    /// input/post norms, `q_norm`/`k_norm`, final `norm`). Qwen3-lineage
    /// drafters ship 1e-6; the GLM-5.3 DFlash2 drafters ship 1e-5. Was a
    /// hardcoded 1e-6 in `from_weights.rs` before this field existed.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// HF `architectures` list. `"DFlash2DraftModel"` marks a DFlash2 drafter,
    /// whose selector/conv fields are then REQUIRED (see [`DflashConfig::validate`]).
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Trained sliding window of the drafter's attention layers, when declared.
    /// NOT consumed by the runtime — the window is CLI-only
    /// (`--dflash-window-size`); the serve compares the two and warns on
    /// mismatch so an operator sees a 4096-vs-2048 drift before measuring.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// HF-style `rope_scaling` block. `None` ⇒ plain RoPE (the v2 2026-04-27
    /// Qwen3.6-DFlash drafter ships `rope_scaling: null`). When present and
    /// `rope_type == "yarn"`, the drafter's YaRN parameters are used to
    /// build the inv_freq table at construction time.
    /// `alias = "rope_parameters"`: newer transformers releases (RadixArk
    /// DSpark, incoai DFlash2) ship the block under that key — without the
    /// alias Atlas silently drops the scaling (the RadixArk config.json had
    /// to be hand-patched before this).
    #[serde(default, alias = "rope_parameters")]
    pub rope_scaling: Option<DflashRopeScaling>,
    /// DSpark Markov head rank. `0` (default) ⇒ plain DFlash drafter with no
    /// Markov head. RadixArk `Qwen3.8-27B-DSpark` ships `markov_rank: 256`
    /// top-level (SpecForge `DSparkConfig` convention — see the checkpoint's
    /// `dspark.py`: DSpark fields are declared as top-level config attrs).
    #[serde(default)]
    pub markov_rank: usize,
    /// DSpark Markov head flavor. Only `"vanilla"` (low-rank learned bigram
    /// bias) is defined by the reference; anything else is rejected at load.
    #[serde(default)]
    pub markov_head_type: Option<String>,
    /// DSpark confidence head (`AcceptRatePredictor`): a `Linear(input, 1)`
    /// predicting per-draft-position acceptance probability, used for
    /// adaptive block length. Loaded when present; consumed by the dynamic-K
    /// scheduling phase (the Markov fixup works without it).
    #[serde(default)]
    pub enable_confidence_head: bool,
    /// When true the confidence head's input is `[hidden ‖ markov_embed]`
    /// (input_dim = hidden_size + markov_rank); when false, hidden only.
    #[serde(default = "default_true")]
    pub confidence_head_with_markov: bool,
}

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

/// HF `architectures` entry that names a DFlash2 drafter.
pub const DFLASH2_ARCHITECTURE: &str = "DFlash2DraftModel";

/// Subset of HF `rope_scaling` block consumed by Atlas. Mirrors the field
/// names in `transformers`' Qwen3 config so `serde_json::from_str` works
/// directly on the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRopeScaling {
    /// Currently only `"yarn"` is recognised; anything else falls back to
    /// plain RoPE with a warning logged at construction time.
    #[serde(default)]
    pub rope_type: Option<String>,
    /// Base RoPE θ when the checkpoint states it INSIDE this block. Newer
    /// transformers releases (`rope_parameters`, incoai DFlash2) put θ here and
    /// ship NO top-level `rope_theta`; without this field the nested value was
    /// dropped and the top-level default (10M) won silently — a 1000× RoPE
    /// error on `incoai/GLM-5.3-Flash-DFlash2` (θ = 10 000). Resolved through
    /// [`DflashConfig::effective_rope_theta`]; never read this field directly.
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<f32>,
}

fn default_block_size() -> usize {
    16
}

/// Nested `dflash_config` block in the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashSubConfig {
    /// Token id used to fill the γ "to-be-predicted" positions during draft
    /// inference. `248070` for Qwen3.6-DFlash.
    pub mask_token_id: u32,
    /// Target-model layer indices to capture intermediate hidden states from.
    /// `[1, 10, 19, 28, 37]` for Qwen3.6-35B-A3B-DFlash. Order matters:
    /// shallow-to-deep concatenation is what `fc` expects.
    pub target_layer_ids: Vec<usize>,
    /// Draft flavor tag. `"dspark"` marks a SpecForge-lineage drafter, whose
    /// row convention is SHIFTED (row j's output = token at position j+1; the
    /// anchor row's output is draft #1) versus the z-lab convention Atlas's
    /// forward was built for (row j predicts at j, row 0 = echo). The runtime
    /// keys the draft-vector rotation off this tag. DFlash2 checkpoints have
    /// no tag and keep the z-lab convention (verified against z-lab
    /// `dflash/model.py::dflash_generate` — anchor-row output discarded,
    /// mask rows fill in place).
    #[serde(default)]
    pub projector_type: Option<String>,

    // ── DFlash2 fields (incoai/z-lab `DFlash2DraftModel`); absent = DFlash1 ──
    /// Two-tap dynamic conv kernel size (2 for DFlash2).
    #[serde(default)]
    pub conv_kernel_size: usize,
    /// Channels per conv group (16 for DFlash2 → hidden/16 groups).
    #[serde(default)]
    pub conv_group_size: usize,
    /// Selector codebook rank (256 for DFlash2).
    #[serde(default)]
    pub selector_rank: usize,
    /// Candidates kept per position for the selector walk (16 for DFlash2).
    #[serde(default)]
    pub selector_top_k: usize,
    /// Block size γ as the drafter was TRAINED, when the checkpoint states it
    /// here. DFlash2 ships `block_size: 8` inside `dflash_config`; DFlash1
    /// ships it top-level only. `None` = not stated, fall back to the
    /// top-level field. Read through [`DflashConfig::effective_block_size`].
    #[serde(default)]
    pub block_size: Option<usize>,
}

impl DflashConfig {
    /// Resolved block size γ: the drafter's own trained value when the
    /// checkpoint states it, else the top-level field.
    ///
    /// The top-level `block_size` defaults to 16, and serde fills that default
    /// happily for a checkpoint that never mentioned it — so a DFlash2 drafter
    /// trained at 8 comes up as 16 unless the sub-config is consulted first.
    /// That is not a cosmetic mismatch: the serve then runs num_drafts=15
    /// against an 8-block drafter, which measured 0% accept on EVERY verify
    /// step, and sizes the drafter's per-sequence pools for twice the block it
    /// will ever use. `--dflash-gamma` still overrides both.
    pub fn effective_block_size(&self) -> usize {
        self.dflash_config
            .as_ref()
            .and_then(|c| c.block_size)
            .unwrap_or(self.block_size)
    }

    /// Resolved base RoPE θ: the nested `rope_parameters.rope_theta` when the
    /// checkpoint states it there, else the top-level field (or its 10M default).
    ///
    /// The nested form wins because a checkpoint that ships it ships ONLY it —
    /// the top-level field is then serde's default, not the drafter's value.
    pub fn effective_rope_theta(&self) -> f32 {
        self.rope_scaling
            .as_ref()
            .and_then(|r| r.rope_theta)
            .unwrap_or(self.rope_theta)
    }

    /// Whether this checkpoint declares itself a DFlash2 drafter.
    ///
    /// Either the HF `architectures` tag OR any DFlash2-only field being set
    /// counts: a checkpoint that ships selector weights under a mislabelled
    /// architecture must still be validated as DFlash2, never silently loaded
    /// as DFlash1 with the selector/conv path disabled.
    pub fn is_dflash2(&self) -> bool {
        self.architectures.iter().any(|a| a == DFLASH2_ARCHITECTURE)
            || self.dflash_config.as_ref().is_some_and(|c| {
                c.conv_kernel_size != 0
                    || c.conv_group_size != 0
                    || c.selector_rank != 0
                    || c.selector_top_k != 0
            })
    }

    /// Fail-closed structural checks, run by `parse_dflash_config` right after
    /// deserialisation so a defective config never reaches the weight loader.
    ///
    /// The DFlash2 fields are `#[serde(default)]` (a DFlash1 checkpoint has
    /// none of them), so a DFlash2 checkpoint missing ONE of them would
    /// otherwise deserialise with a silent 0 — indistinguishable from "not a
    /// DFlash2 drafter" — and `dflash2_active()` would quietly turn the
    /// selector off. Here that is an error with the field named.
    pub fn validate(&self) -> anyhow::Result<()> {
        let sub = match self.dflash_config.as_ref() {
            Some(s) => s,
            None => {
                if self.is_dflash2() {
                    anyhow::bail!(
                        "DFlash2 drafter config declares {DFLASH2_ARCHITECTURE} but has no \
                         `dflash_config` block"
                    );
                }
                return Ok(());
            }
        };
        if sub.target_layer_ids.is_empty() {
            anyhow::bail!("DFlash drafter config: `dflash_config.target_layer_ids` is empty");
        }
        if (sub.mask_token_id as usize) >= self.vocab_size {
            anyhow::bail!(
                "DFlash drafter config: mask_token_id {} is outside vocab_size {}",
                sub.mask_token_id,
                self.vocab_size
            );
        }
        if self.is_dflash2() {
            for (name, v) in [
                ("conv_kernel_size", sub.conv_kernel_size),
                ("conv_group_size", sub.conv_group_size),
                ("selector_rank", sub.selector_rank),
                ("selector_top_k", sub.selector_top_k),
            ] {
                if v == 0 {
                    anyhow::bail!(
                        "DFlash2 drafter config: `dflash_config.{name}` is missing or 0 — \
                         every DFlash2 field is required (fail-closed; a 0 here would \
                         silently disable the selector/conv path)"
                    );
                }
            }
        }
        Ok(())
    }
}
