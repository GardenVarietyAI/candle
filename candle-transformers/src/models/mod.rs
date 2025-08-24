//! Rose-specific candle fork - Qwen2/3 models only
//!
//! This minimal candle fork provides implementations for Qwen2/3 models only.
//! 
//! Supported models:
//!  - [`qwen2`] - Qwen2 language model
//!  - [`qwen3`] - Qwen3 language model
//!  - [`quantized_qwen3`] - Quantized Qwen3 model
//!  - [`qwen3_moe`] - Qwen3 MoE model

pub mod qwen2;
pub mod qwen3;
pub mod qwen3_moe;
pub mod quantized_qwen3;
pub mod with_tracing;
