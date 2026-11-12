//! The feed-forward non-linearity a config names.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::model::Activation;
use crate::runtime::load::family::read::{number, number_or, text};

/// The feed-forward non-linearity the config names, as `hidden_act`,
/// `hidden_activation`, `activation_function` or Nemotron-H's
/// `mlp_hidden_act`; SiLU when it names none.
pub(in crate::runtime::load::family) fn activation(
    raw: &Value,
    model_type: &str,
    path: &Path,
) -> Result<Activation> {
    let mut name = None;
    for key in [
        "hidden_act",
        "hidden_activation",
        "activation_function",
        "mlp_hidden_act",
    ] {
        name = name.or(text(raw, key, path)?);
    }
    match name {
        None | Some("silu") | Some("swish") => Ok(Activation::Silu),
        Some("gelu_pytorch_tanh") | Some("gelu_new") | Some("gelu_fast") => {
            Ok(Activation::GeluTanh)
        }
        Some("gelu") => Ok(Activation::Gelu),
        Some("relu2") => Ok(Activation::Relu2),
        Some("relu") => Ok(Activation::Relu),
        // xIELU's parameters sit in a dense feed-forward's `mlp.act_fn`,
        // which Apertus' dense blocks hold; a mixture's experts, a shared
        // block or a per-layer input hold none.
        Some("xielu") if model_type == "apertus" => Ok(Activation::Xielu),
        Some("xielu") => refused!(
            Unimplemented,
            in raw,
            ["hidden_act"],
            "{} declares hidden_act \"xielu\" for its {model_type} feed-forward; Ster runs xIELU on Apertus' dense feed-forward, whose mlp.act_fn holds its parameters",
            path.display()
        ),
        // Kimi-K3's: `activation_situ_beta`, one when absent or zero, and
        // `activation_situ_linear_beta`, none when absent
        // (`_get_situ_activation_params` in `modeling_kimi_linear.py`).
        Some("situ") => {
            let beta = number_or(raw, "activation_situ_beta", 1.0, path)?;
            Ok(Activation::Situ {
                beta: if beta == 0.0 { 1.0 } else { beta },
                linear_beta: number(raw, "activation_situ_linear_beta", path)?,
            })
        }
        Some(other) => refused!(
            Unimplemented,
            in raw,
            ["hidden_act", "hidden_activation", "activation_function", "mlp_hidden_act"],
            "{} declares hidden_act {other:?} for its {model_type} feed-forward; Ster implements silu, gelu, gelu_pytorch_tanh, gelu_new, gelu_fast, relu, relu2, xielu and situ",
            path.display()
        ),
    }
}
