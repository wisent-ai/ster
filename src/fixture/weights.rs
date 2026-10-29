//! Native Llama checkpoint tensors initialized from the caller's declaration.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

use super::spec::Resolved;

struct Tensors {
    values: HashMap<String, Tensor>,
    rng: StdRng,
}

impl Tensors {
    fn random(
        &mut self,
        name: String,
        rows: usize,
        columns: usize,
        normal: &Normal<f32>,
    ) -> Result<()> {
        let count = rows.checked_mul(columns).with_context(|| {
            format!("fixture tensor {name} shape [{rows}, {columns}] overflows this host")
        })?;
        let mut values = Vec::new();
        values.try_reserve_exact(count).with_context(|| {
            format!("allocate fixture tensor {name} with {count} float elements")
        })?;
        for (index, value) in std::iter::repeat_with(|| normal.sample(&mut self.rng))
            .take(count)
            .enumerate()
        {
            ensure!(
                value.is_finite(),
                "fixture tensor {name} sampled non-finite value {value} at element {index}"
            );
            values.push(value);
        }
        let tensor = Tensor::from_vec(values, (rows, columns), &Device::Cpu)
            .with_context(|| format!("construct fixture tensor {name}"))?;
        self.values.insert(name, tensor);
        Ok(())
    }

    fn vector(&mut self, name: String, width: usize, normal: &Normal<f32>) -> Result<()> {
        let mut values = Vec::new();
        values.try_reserve_exact(width).with_context(|| {
            format!("allocate fixture tensor {name} with {width} float elements")
        })?;
        for (index, value) in std::iter::repeat_with(|| normal.sample(&mut self.rng))
            .take(width)
            .enumerate()
        {
            ensure!(
                value.is_finite(),
                "fixture tensor {name} sampled non-finite value {value} at element {index}"
            );
            values.push(value);
        }
        let tensor = Tensor::from_vec(values, (width,), &Device::Cpu)
            .with_context(|| format!("construct fixture tensor {name}"))?;
        self.values.insert(name, tensor);
        Ok(())
    }

    fn normalization(&mut self, name: String, width: usize, value: f32) -> Result<()> {
        let tensor = Tensor::full(value, (width,), &Device::Cpu)
            .with_context(|| format!("construct fixture normalization tensor {name}"))?;
        self.values.insert(name, tensor);
        Ok(())
    }
}

pub(super) fn build(input: &Resolved) -> Result<HashMap<String, Tensor>> {
    let config = &input.dimensions;
    let initialization = &input.spec.initialization;
    let embedding = initialization
        .embedding
        .distribution("initialization.embedding")?;
    let projection = initialization
        .projection
        .distribution("initialization.projection")?;
    let bias = initialization
        .bias
        .as_ref()
        .map(|value| value.distribution("initialization.bias"))
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
            if with_bias {
                let normal = bias
                    .as_ref()
                    .context("fixture biased projection has no declared initialization.bias")?;
                tensors.vector(format!("{prefix}.{part}.bias"), rows, normal)?;
            }
        }
    }
    Ok(tensors.values)
}
