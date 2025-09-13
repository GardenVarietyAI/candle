use candle_core::{Module, Result, Tensor};
use crate::lora_backend::LoraBackend;

pub struct CpuLoraBackend;

impl CpuLoraBackend {
    pub fn new() -> Self {
        Self
    }
}

impl LoraBackend for CpuLoraBackend {
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
                let a = a.to_device(x.device())?.to_dtype(x.dtype())?;
                let b = b.to_device(x.device())?.to_dtype(x.dtype())?;
                
                let a_t = a.t()?;
                let b_t = b.t()?;

                let dims = x.dims();
                let delta = match dims.len() {
                    2 => {
                        let xa = x.matmul(&a_t)?;
                        xa.matmul(&b_t)?
                    }
                    3 => {
                        let b_dim = dims[0];
                        let l = dims[1];
                        let in_size = dims[2];
                        let bl = b_dim * l;
                        let x2d = x.reshape((bl, in_size))?;
                        let xa = x2d.matmul(&a_t)?;
                        let d2d = xa.matmul(&b_t)?;
                        let out_size = d2d.dim(1)?;
                        d2d.reshape((b_dim, l, out_size))?
                    }
                    _ => {
                        let xa = x.matmul(&a_t)?;
                        xa.matmul(&b_t)?
                    }
                };

                delta.affine(scale, 0.0)?.add(&base_out)
            }
            _ => Ok(base_out),
        }
    }
}