use candle_core::{Device, Result, Tensor};

use crate::lora_cpu::CpuLoraBackend;
use crate::lora_metal::MetalLoraBackend;

pub trait LoraBackend: Send + Sync {
    fn forward(
        &self,
        x: &Tensor,
        base_weight: &Tensor,
        base_bias: Option<&Tensor>,
        lora_a: Option<&Tensor>,
        lora_b: Option<&Tensor>,
        scale: f64,
    ) -> Result<Tensor>;
}

pub fn create_backend(device: &Device) -> Box<dyn LoraBackend> {
    match device {
        Device::Metal(_) => Box::new(MetalLoraBackend::new()),
        _ => Box::new(CpuLoraBackend::new()),
    }
}

#[derive(Debug, Clone)]
pub struct LoraLinearRt {
    weight: Tensor,
    bias: Option<Tensor>,
    lora_a: Option<Tensor>,
    lora_b: Option<Tensor>,
    scale: f64,
    name: String,
}

impl LoraLinearRt {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            weight,
            bias,
            lora_a: None,
            lora_b: None,
            scale: 1.0,
            name: name.into(),
        }
    }

    pub fn set_adapter(&mut self, lora_a: Tensor, lora_b: Tensor, scale: f64) {
        tracing::info!("Setting LoRA adapter {}: A={:?}, B={:?}, scale={}", 
                      self.name, lora_a.dims(), lora_b.dims(), scale);
        self.lora_a = Some(lora_a);
        self.lora_b = Some(lora_b);
        self.scale = scale;
    }

    pub fn clear_adapter(&mut self) {
        self.lora_a = None;
        self.lora_b = None;
        self.scale = 1.0;
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl candle_core::Module for LoraLinearRt {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();
        let backend = create_backend(device);
        backend.forward(
            x,
            &self.weight,
            self.bias.as_ref(),
            self.lora_a.as_ref(),
            self.lora_b.as_ref(),
            self.scale,
        )
    }
}