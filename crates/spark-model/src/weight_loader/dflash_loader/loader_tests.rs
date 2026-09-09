// SPDX-License-Identifier: AGPL-3.0-only

//! Parser tests for the DFlash drafter's `config.json`.
//!
//! Split from `dflash_loader.rs` for the 500-LoC cap, which the file crossed
//! when the DFlash2 sub-config gained its own block-size resolution. Exact
//! piecewise copy — no test changed in the move.

use super::*;

const SHIPPED_CONFIG: &str = r#"{
    "hidden_size": 2048,
    "num_hidden_layers": 8,
    "intermediate_size": 6144,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 248320,
    "draft_vocab_size": 248320,
    "tie_word_embeddings": false,
    "block_size": 16,
    "rope_theta": 10000000.0,
    "rope_scaling": null,
    "dflash_config": {
        "mask_token_id": 248070,
        "target_layer_ids": [1, 10, 19, 28, 37]
    }
}"#;

#[test]
fn shipped_qwen3_6_fields_reach_the_runtime_config() {
    let config = parse_dflash_config(SHIPPED_CONFIG).expect("parse drafter config");
    assert_eq!(config.num_hidden_layers, 8);
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.intermediate_size, 6144);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 4);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.vocab_size, 248320);
    assert_eq!(config.draft_vocab_size, Some(248320));
    assert!(!config.tie_word_embeddings);
    assert_eq!(config.block_size, 16);
    assert_eq!(config.rope_theta, 10_000_000.0);
    assert!(config.rope_scaling.is_none());
    let sub = config.dflash_config.expect("dflash_config present");
    assert_eq!(sub.mask_token_id, 248070);
    assert_eq!(sub.target_layer_ids, vec![1, 10, 19, 28, 37]);
}

#[test]
fn omitted_optional_fields_use_runtime_defaults() {
    let config = parse_dflash_config(
        r#"{
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 256
        }"#,
    )
    .unwrap();

    assert_eq!(config.block_size, 16);
    assert_eq!(config.rope_theta, 10_000_000.0);
    assert!(!config.tie_word_embeddings);
    assert!(config.draft_vocab_size.is_none());
    assert!(config.dflash_config.is_none());
    assert!(config.rope_scaling.is_none());
}

#[test]
fn malformed_runtime_field_is_rejected_with_parser_context() {
    let malformed =
        SHIPPED_CONFIG.replace("\"mask_token_id\": 248070", "\"mask_token_id\": \"bad\"");
    let error = parse_dflash_config(&malformed).unwrap_err();
    assert_eq!(error.to_string(), "Parsing DFlash drafter config.json");
    assert!(format!("{error:#}").contains("invalid type: string \"bad\""));
}

/// A DFlash2 checkpoint states its trained block size INSIDE
/// `dflash_config`; the top-level field is absent and serde fills its
/// default of 16. Resolving from the top level alone therefore runs an
/// 8-block drafter at gamma=16 -- 0% accept on every verify step, and
/// drafter pools sized for twice the block. Hermetic: parses a literal,
/// no checkpoint needed.
#[test]
fn effective_block_size_prefers_the_drafters_own_value() {
    let json = r#"{
        "hidden_size": 5120, "num_hidden_layers": 5,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 17408, "vocab_size": 248320, "head_dim": 128,
        "dflash_config": {
            "block_size": 8, "mask_token_id": 248070,
            "target_layer_ids": [1, 10, 19, 28, 37]
        }
    }"#;
    let cfg = parse_dflash_config(json).expect("parses");
    assert_eq!(cfg.block_size, 16, "top-level default is still 16");
    assert_eq!(
        cfg.effective_block_size(),
        8,
        "the drafter's own block_size must win over the top-level default"
    );
}

/// A DFlash1 checkpoint states it top-level only: nothing to prefer, so
/// the resolved value is unchanged from before this resolver existed.
#[test]
fn effective_block_size_falls_back_to_the_top_level() {
    let json = r#"{
        "hidden_size": 2048, "num_hidden_layers": 8,
        "num_attention_heads": 32, "num_key_value_heads": 4,
        "intermediate_size": 6144, "vocab_size": 248320, "head_dim": 128,
        "block_size": 16
    }"#;
    let cfg = parse_dflash_config(json).expect("parses");
    assert_eq!(cfg.effective_block_size(), 16);
}

// ── incoai/GLM-5.3-Flash-DFlash2 (config.json read live from the Hub, 2026-09-06) ──
//
// Everything the Qwen fixture above coincidentally shares with Atlas's old
// defaults, this one does not: θ is NESTED (10 000, not the 10M default),
// rms_norm_eps is 1e-5 (not the hardcoded 1e-6), the block size lives in
// `dflash_config`, and the four DFlash2 fields are all present.
const INCOAI_GLM53_FLASH_DFLASH2_CONFIG: &str = r#"{
  "architectures": ["DFlash2DraftModel"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": null,
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 154856,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 14, 24, 33, 42]
  },
  "dtype": "bfloat16",
  "eos_token_id": [154820, 154827, 154829],
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "is_causal": false,
  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
  "max_position_embeddings": 1048576,
  "max_window_layers": 5,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 45,
  "pad_token_id": 154820,
  "rms_norm_eps": 1e-05,
  "rope_parameters": {
    "rope_theta": 10000.0,
    "rope_type": "default"
  },
  "sliding_window": 2048,
  "tie_word_embeddings": false,
  "transformers_version": "5.7.0",
  "use_cache": false,
  "use_sliding_window": true,
  "vocab_size": 154880
}"#;

#[test]
fn glm53_flash_dflash2_nested_rope_theta_wins_over_the_default() {
    let c = parse_dflash_config(INCOAI_GLM53_FLASH_DFLASH2_CONFIG).expect("parse");
    // The top-level field is serde's default — the checkpoint never set it.
    assert_eq!(c.rope_theta, 10_000_000.0);
    // The resolved value is the nested one.
    assert_eq!(c.effective_rope_theta(), 10_000.0);
    let rs = c
        .rope_scaling
        .as_ref()
        .expect("rope_parameters aliased onto rope_scaling");
    assert_eq!(rs.rope_type.as_deref(), Some("default"));
    assert_eq!(rs.rope_theta, Some(10_000.0));
}

#[test]
fn glm53_flash_dflash2_rms_norm_eps_and_shape_reach_the_runtime() {
    let c = parse_dflash_config(INCOAI_GLM53_FLASH_DFLASH2_CONFIG).expect("parse");
    assert_eq!(c.rms_norm_eps, 1e-5);
    assert_eq!(c.hidden_size, 4096);
    assert_eq!(c.intermediate_size, 12288);
    assert_eq!(c.num_hidden_layers, 5);
    assert_eq!(c.num_attention_heads, 32);
    assert_eq!(c.num_key_value_heads, 8);
    assert_eq!(c.head_dim, 128);
    assert_eq!(c.vocab_size, 154880);
    assert_eq!(c.sliding_window, Some(2048));
    assert_eq!(
        c.effective_block_size(),
        8,
        "block size is NESTED; top-level default 16 must not win"
    );
    let sub = c.dflash_config.as_ref().unwrap();
    assert_eq!(sub.mask_token_id, 154856);
    assert_eq!(sub.target_layer_ids, vec![5, 14, 24, 33, 42]);
    assert_eq!(sub.conv_kernel_size, 2);
    assert_eq!(sub.conv_group_size, 16);
    assert_eq!(sub.selector_rank, 256);
    assert_eq!(sub.selector_top_k, 16);
    assert!(c.is_dflash2());
}

#[test]
fn qwen_dflash1_fixture_keeps_its_old_defaults() {
    // The certified drafter must be unaffected: top-level θ, eps 1e-6, DFlash1.
    let c = parse_dflash_config(SHIPPED_CONFIG).expect("parse");
    assert_eq!(c.effective_rope_theta(), 10_000_000.0);
    assert_eq!(c.rms_norm_eps, 1e-6);
    assert!(!c.is_dflash2());
    assert!(c.validate().is_ok());
}

#[test]
fn nested_rope_theta_beats_top_level_when_both_are_present() {
    let c = parse_dflash_config(
        r#"{
            "hidden_size": 64, "num_hidden_layers": 1, "intermediate_size": 128,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "vocab_size": 256, "rope_theta": 5.0,
            "rope_parameters": {"rope_theta": 7.0, "rope_type": "default"}
        }"#,
    )
    .expect("parse");
    assert_eq!(c.effective_rope_theta(), 7.0);
}

/// Fail-closed: a DFlash2 checkpoint missing ONE selector/conv field used to
/// deserialise with a silent 0 (indistinguishable from "not DFlash2") and
/// quietly disable the selector. Now it is a parse error naming the field.
#[test]
fn dflash2_missing_selector_field_is_a_parse_error() {
    let broken = INCOAI_GLM53_FLASH_DFLASH2_CONFIG.replace(r#""selector_top_k": 16,"#, "");
    assert!(
        !broken.contains("selector_top_k"),
        "fixture edit must remove the key"
    );
    let err = parse_dflash_config(&broken).expect_err("must fail closed");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("selector_top_k"),
        "error must name the field: {msg}"
    );
}

#[test]
fn dflash2_architecture_without_dflash_config_block_is_rejected() {
    let err = parse_dflash_config(
        r#"{
            "architectures": ["DFlash2DraftModel"],
            "hidden_size": 64, "num_hidden_layers": 1, "intermediate_size": 128,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "vocab_size": 256
        }"#,
    )
    .expect_err("must fail closed");
    assert!(format!("{err:#}").contains("dflash_config"));
}

#[test]
fn mask_token_outside_vocab_is_rejected() {
    let broken = INCOAI_GLM53_FLASH_DFLASH2_CONFIG
        .replace(r#""vocab_size": 154880"#, r#""vocab_size": 154856"#);
    let err =
        parse_dflash_config(&broken).expect_err("mask_token_id == vocab_size is out of range");
    assert!(format!("{err:#}").contains("mask_token_id"));
}
