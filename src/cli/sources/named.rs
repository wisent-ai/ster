//! The sources the import plan names that no Ster command reads (IMP-STER-<NN>), each
//! listed by `ster sources` with why and what to do, and answered by `discover` as such.

use serde_json::{Value, json};

const VECTORS: &str = "keeps steering vectors in its own Python objects, which no Ster command reads";
const AS_PAIRS: &str = "make the contrastive pairs the vectors came from a pair set (ster pairs import) and train the vector in Ster";
const WEIGHTS: &str = "keeps weights in another layout than the safetensors --model reads";
const AS_SAFETENSORS: &str = "convert it to a Transformers directory with safetensors weights and use IMP-STER-12";

pub(super) const NAMED: &[(&str, &str, &str, &str)] = &[
    ("IMP-STER-02", "local Parquet or Arrow files", "are columnar, and Ster reads JSON and JSON Lines", "write the rows out as JSON Lines (datasets' to_json) and read them as IMP-STER-01 or 06"),
    ("IMP-STER-07", "repeng control vectors", VECTORS, AS_PAIRS),
    ("IMP-STER-08", "steering-vectors", VECTORS, AS_PAIRS),
    ("IMP-STER-09", "TransformerLens hooks", VECTORS, AS_PAIRS),
    ("IMP-STER-10", "nnsight interventions", VECTORS, AS_PAIRS),
    ("IMP-STER-11", "SAE Lens features", "are sparse-autoencoder features, which no Ster command reads", AS_PAIRS),
    ("IMP-STER-14", "an MLX model", WEIGHTS, AS_SAFETENSORS),
    ("IMP-STER-15", "a PyTorch checkpoint (.bin, .pt)", WEIGHTS, AS_SAFETENSORS),
    ("IMP-STER-18", "an MLX LoRA adapter", "is an adapter in MLX's own layout, which no Ster command reads", "merge it into its base model, convert that to safetensors and use it as IMP-STER-12"),
    ("IMP-STER-17", "a PEFT/LoRA adapter", "is a layout Ster writes as a tune artifact but does not read back", "merge it into its base model and use the merged directory as IMP-STER-12"),
];

pub(super) fn row(&(id, source, why, remedy): &(&str, &str, &str, &str)) -> Value {
    json!({"adapter": id, "source": source, "kind": "D", "found": false, "why": why, "remedy": remedy})
}
