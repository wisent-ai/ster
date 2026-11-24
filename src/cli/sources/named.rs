//! The sources the import plan names that no Ster command reads, each listed by `ster
//! sources` under its name with why and what to do, and answered by `discover` as such.

use serde_json::{Value, json};

const VECTORS: &str = "keeps steering vectors in its own Python objects, which no Ster command reads";
const AS_PAIRS: &str = "make the contrastive pairs the vectors came from a pair set (ster pairs import) and train the vector in Ster";
const WEIGHTS: &str = "keeps weights in another layout than the safetensors --model reads";
const AS_SAFETENSORS: &str = "convert it to a Transformers directory with safetensors weights and use it as transformers-checkpoint";

pub(super) const NAMED: &[(&str, &str, &str, &str)] = &[
    ("parquet-or-arrow", "local Parquet or Arrow files", "are columnar, and Ster reads JSON and JSON Lines", "write the rows out as JSON Lines (datasets' to_json) and read them as hugging-face-preferences or dpo"),
    ("repeng", "repeng control vectors", VECTORS, AS_PAIRS),
    ("steering-vectors", "steering-vectors", VECTORS, AS_PAIRS),
    ("transformerlens", "TransformerLens hooks", VECTORS, AS_PAIRS),
    ("nnsight", "nnsight interventions", VECTORS, AS_PAIRS),
    ("sae-lens", "SAE Lens features", "are sparse-autoencoder features, which no Ster command reads", AS_PAIRS),
    ("mlx-model", "an MLX model", WEIGHTS, AS_SAFETENSORS),
    ("pytorch-checkpoint", "a PyTorch checkpoint (.bin, .pt)", WEIGHTS, AS_SAFETENSORS),
    ("mlx-lora-adapter", "an MLX LoRA adapter", "is an adapter in MLX's own layout, which no Ster command reads", "merge it into its base model, convert that to safetensors and use it as transformers-checkpoint"),
    ("peft-lora-adapter", "a PEFT/LoRA adapter", "is a layout Ster writes as a tune artifact but does not read back", "merge it into its base model and use the merged directory as transformers-checkpoint"),
    ("wisent-evaluators", "wisent-evaluators evaluations", "name the evaluator, verdict, score, details and meta of each request, not the prompt or the response they judged (wisent-evaluators' evaluation format), so an evaluation alone gives no pair", "for a prompt with a truthful and an untruthful response, write the row (prompt, chosen: the truthful one, rejected: the untruthful one) and read it as dpo"),
];

pub(super) fn row(&(name, source, why, remedy): &(&str, &str, &str, &str)) -> Value {
    json!({"adapter": name, "source": source, "kind": "D", "found": false, "why": why, "remedy": remedy})
}
