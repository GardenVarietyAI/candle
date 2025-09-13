use candle_core::{Module, Result, Tensor};
use crate::lora_backend::LoraBackend;

pub struct MetalLoraBackend;

impl MetalLoraBackend {
    pub fn new() -> Self {
        Self
    }
}

impl LoraBackend for MetalLoraBackend {
    fn forward(
        &self,
        x: &Tensor,
        base_weight: &Tensor,
        base_bias: Option<&Tensor>,
        lora_a: Option<&Tensor>,
        lora_b: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor> {
        let linear = candle_nn::Linear::new(base_weight.clone(), base_bias.cloned());
        let base_out = linear.forward(x)?;

        match (lora_a, lora_b) {
            (Some(a), Some(b)) => {
                let a = a.to_device(x.device())?.to_dtype(x.dtype())?.contiguous()?;
                let b = b.to_device(x.device())?.to_dtype(x.dtype())?.contiguous()?;
                
                let a_t = a.t()?.contiguous()?;
                let b_t = b.t()?.contiguous()?;

                let dims = x.dims();
                let delta = match dims.len() {
                    2 => {
                        let xa = x.contiguous()?.matmul(&a_t).map_err(|e| {
                            candle_core::Error::Msg(format!(
                                "Metal LoRA x @ A.T failed: x={:?} A.T={:?} error={}",
                                x.dims(), a_t.dims(), e
                            ))
                        })?;
                        xa.matmul(&b_t).map_err(|e| {
                            candle_core::Error::Msg(format!(
                                "Metal LoRA xa @ B.T failed: xa={:?} B.T={:?} error={}",
                                xa.dims(), b_t.dims(), e
                            ))
                        })?
                    }
                    3 => {
                        let b_dim = dims[0];
                        let l = dims[1];
                        let in_size = dims[2];
                        let bl = b_dim * l;
                        let x2d = x.contiguous()?.reshape((bl, in_size))?;
                        let xa = x2d.matmul(&a_t).map_err(|e| {
                            candle_core::Error::Msg(format!(
                                "Metal LoRA x2d @ A.T failed: x2d={:?} A.T={:?} error={}",
                                x2d.dims(), a_t.dims(), e
                            ))
                        })?;
                        let d2d = xa.matmul(&b_t).map_err(|e| {
                            candle_core::Error::Msg(format!(
                                "Metal LoRA xa @ B.T failed: xa={:?} B.T={:?} error={}",
                                xa.dims(), b_t.dims(), e
                            ))
                        })?;
                        let out_size = d2d.dim(1)?;
                        d2d.reshape((b_dim, l, out_size))?
                    }
                    _ => {
                        let xa = x.contiguous()?.matmul(&a_t)?;
                        xa.matmul(&b_t)?
                    }
                };

                delta.affine(scale, 0.0)?.add(&base_out)
            }
            _ => Ok(base_out),
        }
    }
}