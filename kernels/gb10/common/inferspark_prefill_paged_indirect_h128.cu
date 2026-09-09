// SPDX-License-Identifier: AGPL-3.0-only
//
// Paged Prefill Flash Attention, INDIRECT scalar args — HDIM=128 specialization.
//
// Same kernel body as inferspark_prefill_paged_indirect.cu (HDIM=256 default)
// compiled with HDIM=128 under a distinct symbol. Selected by
// dflash_head::attn_width for drafters whose head_dim is 128 (the DFlash2
// GLM-5.3-Flash and Qwen3.6 drafters). Running such a drafter through the
// HDIM=256 build loads heads h and h+1 into one 256-wide Q/K tile, so every
// head's softmax scores become s(h)+s(h+1) — in-bounds output, wrong weights
// (A93, spark-bench HANDOFF-08, 2026-09-07: count_pin acceptance 4.55/7 → 6.17/7
// with this module alone). The target model's own prefill kernels are untouched.
#define HDIM 128
#define KERNEL_NAME inferspark_prefill_paged_indirect_h128
#include "inferspark_prefill_paged_indirect.cu"
