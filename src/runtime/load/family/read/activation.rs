//! The feed-forward non-linearity a config names.

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::Activation;
use crate::runtime::load::family::read::{number, text};

/// The feed-forward non-linearity the config names, as `hidden_act`,
/// `hidden_activation`, `activation_function` or Nemotron-H's
/// `mlp_hidden_act`; SiLU when it names none.
pub(in crate::runtime::load::family) fn activation(
    raw: &Value,
    model_type: &str,
    path: &Path,
) -> Result<Activation> {
    let name = text(raw, "hidden_act")
        .or_else(|| text(raw, "hidden_activation"))
        .or_else(|| text(raw, "activation_function"))
        .or_else(|| text(raw, "mlp_hidden_act"));
    match name {
        None | Some("silu") | Some("swish") => Ok(Activation::Silu),
        Some("gelu_pytorch_tanh") | Some("gelu_new") | Some("gelu_fast") => {
            Ok(Activation::GeluTanh)
        }
        Some("gelu") => Ok(Activation::Gelu),
        Some("relu2") => Ok(Activation::Relu2),
        Some("relu") => Ok(Activation::Relu),
        Some("xielu") => Ok(Activation::Xielu),
        // Kimi-K3's: `activation_situ_beta`, one when absent or zero, and
        // `activation_situ_linear_beta`, none when absent
        // (`_get_situ_activation_params` in `modeling_kimi_linear.py`).
        Some("situ") => Ok(Activation::Situ {
            beta: number(raw, "activation_situ_beta")
                .filter(|beta| *beta != 0.0)
                .unwrap_or(1.0),
            linear_beta: number(raw, "activation_situ_linear_beta"),
        }),
        Some(other) => bail!(
            "{} declares hidden_act {other:?} for its {model_type} feed-forward; Ster implements silu, gelu, gelu_pytorch_tanh, gelu_new, gelu_fast, relu, relu2, xielu and situ",
            path.display()
        ),
    }
}
