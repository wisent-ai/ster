//! Native Llama checkpoint tensors initialized from the caller's declaration.

use std::collections::HashMap;

use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

use crate::failures::{
    fixture::{
        declaration::BIAS_MISSING,
        output::{SAMPLE_NOT_FINITE, TENSOR_UNBUILT},
    },
    refuse,
};

use super::spec::Resolved;

/// The refusal of tensor `name` of `shape` and `count` values this host
/// cannot build, the allocator's answer `error`.
fn unbuilt(name: &str, shape: &str, count: &str, error: &dyn std::fmt::Display) -> anyhow::Error {
    refuse(
        &TENSOR_UNBUILT,
        &[
            ("name", name),
            ("shape", shape),
            ("count", count),
            ("error", &error.to_string()),
        ],
    )
}

struct Tensors {
    values: HashMap<String, Tensor>,
    rng: StdRng,
}

impl Tensors {
    /// `count` values drawn from `normal` for tensor `name` of `shape`.
    fn drawn(&mut self, name: &str, shape: &str, count: usize, normal: &Normal<f32>) -> Result<Vec<f32>> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|error| unbuilt(name, shape, &count.to_string(), &error))?;
        for (index, value) in std::iter::repeat_with(|| normal.sample(&mut self.rng))
            .take(count)
            .enumerate()
        {
            if !value.is_finite() {
                return Err(refuse(
                    &SAMPLE_NOT_FINITE,
                    &[
                        ("name", name),
                        ("shape", shape),
                        ("index", &index.to_string()),
                        ("value", &value.to_string()),
                    ],
                ));
            }
            values.push(value);
        }
        Ok(values)
    }

    fn random(
        &mut self,
        name: String,
        rows: usize,
        columns: usize,
        normal: &Normal<f32>,
    ) -> Result<()> {
        let shape = format!("[{rows}, {columns}]");
        let Some(count) = rows.checked_mul(columns) else {
            let count = format!("{rows} × {columns}, past {}", usize::MAX);
            return Err(unbuilt(&name, &shape, &count, &"the count overflows this host's sizes"));
        };
        let values = self.drawn(&name, &shape, count, normal)?;
        let tensor = Tensor::from_vec(values, (rows, columns), &Device::Cpu)
            .map_err(|error| unbuilt(&name, &shape, &count.to_string(), &error))?;
        self.values.insert(name, tensor);
        Ok(())
    }

    fn vector(&mut self, name: String, width: usize, normal: &Normal<f32>) -> Result<()> {
        let shape = format!("[{width}]");
        let values = self.drawn(&name, &shape, width, normal)?;
        let tensor = Tensor::from_vec(values, (width,), &Device::Cpu)
            .map_err(|error| unbuilt(&name, &shape, &width.to_string(), &error))?;
        self.values.insert(name, tensor);
        Ok(())
    }

    fn normalization(&mut self, name: String, width: usize, value: f32) -> Result<()> {
        let tensor = Tensor::full(value, (width,), &Device::Cpu)
            .map_err(|error| unbuilt(&name, &format!("[{width}]"), &width.to_string(), &error))?;
        self.values.insert(name, tensor);
        Ok(())
    }
}

pub(super) fn build(input: &Resolved) -> Result<HashMap<String, Tensor>> {
    let config = &input.dimensions;
    let initialization = &input.spec.initialization;
    let path = input.declaration.as_path();
    let embedding = initialization
        .embedding
        .distribution("initialization.embedding", path)?;
    let projection = initialization
        .projection
        .distribution("initialization.projection", path)?;
    let bias = initialization
        .bias
        .as_ref()
        .map(|value| value.distribution("initialization.bias", path))
        .transpose()?;
    let hidden = config.hidden_size.get();
    let intermediate = config.intermediate_size.get();
    let mut tensors = Tensors {
        values: HashMap::new(),
        rng: StdRng::seed_from_u64(initialization.seed),
    };
    // Tensor paths are native checkpoint wire identities, not a behavioral
    // word classifier; retain them as that protocol requires:
    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    tensors.random(
        "model.embed_tokens.weight".into(),
        config.vocab_size.get(),
        hidden,
        &embedding,
    )?;
    if !config.tie_word_embeddings {
        tensors.random(
            "lm_head.weight".into(),
            config.vocab_size.get(),
            hidden,
            &embedding,
        )?;
    }
    tensors.normalization(
        "model.norm.weight".into(),
        hidden,
        initialization.normalization,
    )?;
    // Native ModuleList checkpoint layer identities are zero based:
    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    for layer in 0..config.num_hidden_layers.get() {
        let prefix = format!("model.layers.{layer}");
        tensors.normalization(
            format!("{prefix}.input_layernorm.weight"),
            hidden,
            initialization.normalization,
        )?;
        tensors.normalization(
            format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
            initialization.normalization,
        )?;
        for (part, rows, columns, with_bias) in [
            (
                "self_attn.q_proj",
                input.query_width,
                hidden,
                config.attention_bias,
            ),
            (
                "self_attn.k_proj",
                input.kv_width,
                hidden,
                config.attention_bias,
            ),
            (
                "self_attn.v_proj",
                input.kv_width,
                hidden,
                config.attention_bias,
            ),
            (
                "self_attn.o_proj",
                hidden,
                input.query_width,
                config.attention_bias,
            ),
            ("mlp.gate_proj", intermediate, hidden, false),
            ("mlp.up_proj", intermediate, hidden, false),
            ("mlp.down_proj", hidden, intermediate, false),
        ] {
            tensors.random(
                format!("{prefix}.{part}.weight"),
                rows,
                columns,
                &projection,
            )?;
            match (with_bias, bias.as_ref()) {
                (true, Some(normal)) => {
                    tensors.vector(format!("{prefix}.{part}.bias"), rows, normal)?;
                }
                (true, None) => {
                    let shown = path.display().to_string();
                    let config = format!("the config of {shown} (writing {prefix}.{part}.bias)");
                    return Err(refuse(&BIAS_MISSING, &[("path", &shown), ("config", &config)]));
                }
                (false, _) => {}
            }
        }
    }
    Ok(tensors.values)
}
