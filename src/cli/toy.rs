//! `ster toy-model <DIR>`: a tiny offline Llama-family checkpoint that Ster
//! can load, so every command can be exercised end to end on a laptop with no
//! download, no GPU and no account.
//!
//! The checkpoint is real in shape only: `config.json` says
//! `model_type: "llama"`, `tokenizer.json` is a WordLevel tokenizer over a
//! ~60-word vocabulary, and `model.safetensors` holds seeded random weights in
//! the exact tensor layout Ster's decoder loads. Generated text is
//! deterministic gibberish drawn from the toy vocabulary; the point is the
//! mechanics, not the prose.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Map, Value};

/// Dimensions of the toy: the smallest grouped-query Llama shape (fewer KV
/// heads than query heads) that exercises every path Ster's decoder has.
const HIDDEN: usize = 64;
const INTERMEDIATE: usize = 128;
const LAYERS: usize = 4;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const MAX_POSITIONS: usize = 256;
/// Llama's own defaults for the norm epsilon and the rotary base.
const RMS_NORM_EPS: f64 = 1e-5;
const ROPE_THETA: f64 = 10_000.0;
/// The same seed on every run, so two toy checkpoints are identical.
const SEED: u64 = 7;
/// Standard deviations of the random weights: embeddings small, projections
/// larger, so activations neither vanish nor explode across four layers.
const EMBEDDING_SCALE: f32 = 0.02;
const PROJECTION_SCALE: f32 = 0.05;
/// Unknown, beginning and end of sequence; their ids are their positions.
const SPECIALS: [&str; 3] = ["[UNK]", "<s>", "</s>"];
const WORDS: &str = "the sea is calm and quiet tonight storm waves crash loud against rocks \
    wind howls water lies still air gentle harbor boat rests at anchor night sky clear dark \
    thunder rolls over hills morning light soft warm cold rain falls hard fast slow breeze \
    drifts a in on answer question describe evening lake surface mirror like broken churns \
    white foam . , : ?";

type Tensor = (String, Vec<usize>, Vec<f32>);

/// One draw from a normal distribution (Box-Muller), scaled.
fn gauss(rng: &mut StdRng, scale: f32) -> f32 {
    let u1: f32 = rng.random_range(f32::EPSILON..1.0);
    let u2: f32 = rng.random();
    scale * (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn random(rng: &mut StdRng, name: String, shape: &[usize], scale: f32) -> Tensor {
    let count = shape.iter().product();
    let values = (0..count).map(|_| gauss(rng, scale)).collect();
    (name, shape.to_vec(), values)
}

fn ones(name: String, size: usize) -> Tensor {
    (name, vec![size], vec![1.0; size])
}

fn write_safetensors(path: &Path, tensors: &[Tensor]) -> Result<()> {
    let mut header = Map::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = payload.len();
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]}),
        );
    }
    let header = serde_json::to_vec(&Value::Object(header))?;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&payload);
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

fn special(token: &str) -> usize {
    SPECIALS.iter().position(|s| *s == token).expect("a declared special token")
}

fn config(vocab_size: usize) -> Value {
    let mut config = Map::new();
    config.insert("model_type".into(), json!("llama"));
    config.insert("hidden_size".into(), json!(HIDDEN));
    config.insert("intermediate_size".into(), json!(INTERMEDIATE));
    config.insert("vocab_size".into(), json!(vocab_size));
    config.insert("num_hidden_layers".into(), json!(LAYERS));
    config.insert("num_attention_heads".into(), json!(HEADS));
    config.insert("num_key_value_heads".into(), json!(KV_HEADS));
    config.insert("rms_norm_eps".into(), json!(RMS_NORM_EPS));
    config.insert("rope_theta".into(), json!(ROPE_THETA));
    config.insert("bos_token_id".into(), json!(special("<s>")));
    config.insert("eos_token_id".into(), json!(special("</s>")));
    config.insert("max_position_embeddings".into(), json!(MAX_POSITIONS));
    config.insert("tie_word_embeddings".into(), json!(false));
    Value::Object(config)
}

fn tokenizer(tokens: &[&str]) -> Value {
    let vocab: Map<String, Value> = tokens
        .iter()
        .enumerate()
        .map(|(index, token)| (token.to_string(), json!(index)))
        .collect();
    let added: Vec<Value> = SPECIALS
        .iter()
        .map(|token| {
            json!({"id": special(token), "content": token, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false, "special": true})
        })
        .collect();
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": SPECIALS[0]},
    })
}

fn weights(vocab_size: usize) -> Vec<Tensor> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let kv_width = HIDDEN / HEADS * KV_HEADS;
    let mut tensors = vec![
        random(&mut rng, "model.embed_tokens.weight".into(), &[vocab_size, HIDDEN], EMBEDDING_SCALE),
        random(&mut rng, "lm_head.weight".into(), &[vocab_size, HIDDEN], EMBEDDING_SCALE),
        ones("model.norm.weight".into(), HIDDEN),
    ];
    for layer in 0..LAYERS {
        let p = format!("model.layers.{layer}");
        tensors.push(ones(format!("{p}.input_layernorm.weight"), HIDDEN));
        tensors.push(ones(format!("{p}.post_attention_layernorm.weight"), HIDDEN));
        for (part, rows, cols) in [
            ("self_attn.q_proj", HIDDEN, HIDDEN),
            ("self_attn.k_proj", kv_width, HIDDEN),
            ("self_attn.v_proj", kv_width, HIDDEN),
            ("self_attn.o_proj", HIDDEN, HIDDEN),
            ("mlp.gate_proj", INTERMEDIATE, HIDDEN),
            ("mlp.up_proj", INTERMEDIATE, HIDDEN),
            ("mlp.down_proj", HIDDEN, INTERMEDIATE),
        ] {
            let name = format!("{p}.{part}.weight");
            tensors.push(random(&mut rng, name, &[rows, cols], PROJECTION_SCALE));
        }
    }
    tensors
}

pub(super) fn run(out: &Path) -> Result<()> {
    fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let tokens: Vec<&str> = SPECIALS.iter().copied().chain(WORDS.split_whitespace()).collect();
    let pretty = |value: &Value| serde_json::to_string_pretty(value).map(|text| text + "\n");
    fs::write(out.join("config.json"), pretty(&config(tokens.len()))?)?;
    fs::write(out.join("tokenizer.json"), pretty(&tokenizer(&tokens))?)?;
    write_safetensors(&out.join("model.safetensors"), &weights(tokens.len()))?;
    println!("{}", out.display());
    Ok(())
}
