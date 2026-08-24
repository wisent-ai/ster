#!/usr/bin/env python3
"""Build a tiny offline Llama-family checkpoint that Ster can load.

The checkpoint is real in shape only: `config.json` says `model_type: "llama"`,
`tokenizer.json` is a WordLevel tokenizer over a ~60-word vocabulary, and
`model.safetensors` holds seeded random weights in the exact tensor layout
Ster's decoder loads (`model.embed_tokens.weight`, per-layer `self_attn.*`,
`mlp.*`, norms, `lm_head.weight`). It exists so every Ster command can be
exercised end to end on a laptop with no download, no GPU, and no account.
Generated text is deterministic gibberish drawn from the toy vocabulary;
the point is the mechanics, not the prose.

Usage: python3 make-toy-model.py <output-directory>

Stdlib only. Dimensions: hidden_size 64, 4 layers, 4 heads (2 KV heads),
intermediate_size 128, max_position_embeddings 256.
"""

import json
import random
import struct
import sys
from pathlib import Path

HIDDEN = 64
INTERMEDIATE = 128
LAYERS = 4
HEADS = 4
KV_HEADS = 2
MAX_POSITIONS = 256

WORDS = [
    "the", "sea", "is", "calm", "and", "quiet", "tonight", "storm", "waves",
    "crash", "loud", "against", "rocks", "wind", "howls", "water", "lies",
    "still", "air", "gentle", "harbor", "boat", "rests", "at", "anchor",
    "night", "sky", "clear", "dark", "thunder", "rolls", "over", "hills",
    "morning", "light", "soft", "warm", "cold", "rain", "falls", "hard",
    "fast", "slow", "breeze", "drifts", "a", "in", "on", "answer", "question",
    "describe", "evening", "lake", "surface", "mirror", "like", "broken",
    "churns", "white", "foam", ".", ",", ":", "?",
]


def write_safetensors(path: Path, tensors: list[tuple[str, list[int], list[float]]]) -> None:
    header: dict[str, dict] = {}
    payload = bytearray()
    offset = 0
    for name, shape, values in tensors:
        data = struct.pack(f"<{len(values)}f", *values)
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        payload += data
        offset += len(data)
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: make-toy-model.py <output-directory>")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    specials = ["[UNK]", "<s>", "</s>"]
    vocab = {token: index for index, token in enumerate(specials + WORDS)}
    vocab_size = len(vocab)

    (out / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": vocab_size,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_position_embeddings": MAX_POSITIONS,
        "tie_word_embeddings": False,
    }, indent=2) + "\n")

    (out / "tokenizer.json").write_text(json.dumps({
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": vocab[token], "content": token, "single_word": False,
             "lstrip": False, "rstrip": False, "normalized": False,
             "special": True}
            for token in specials
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    }, indent=2) + "\n")

    rng = random.Random(7)

    def tensor(name: str, *shape: int, scale: float) -> tuple[str, list[int], list[float]]:
        count = 1
        for dim in shape:
            count *= dim
        return name, list(shape), [rng.gauss(0.0, scale) for _ in range(count)]

    def ones(name: str, size: int) -> tuple[str, list[int], list[float]]:
        return name, [size], [1.0] * size

    kv_width = HIDDEN // HEADS * KV_HEADS
    tensors = [
        tensor("model.embed_tokens.weight", vocab_size, HIDDEN, scale=0.02),
        tensor("lm_head.weight", vocab_size, HIDDEN, scale=0.02),
        ones("model.norm.weight", HIDDEN),
    ]
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}"
        tensors += [
            ones(f"{prefix}.input_layernorm.weight", HIDDEN),
            ones(f"{prefix}.post_attention_layernorm.weight", HIDDEN),
            tensor(f"{prefix}.self_attn.q_proj.weight", HIDDEN, HIDDEN, scale=0.05),
            tensor(f"{prefix}.self_attn.k_proj.weight", kv_width, HIDDEN, scale=0.05),
            tensor(f"{prefix}.self_attn.v_proj.weight", kv_width, HIDDEN, scale=0.05),
            tensor(f"{prefix}.self_attn.o_proj.weight", HIDDEN, HIDDEN, scale=0.05),
            tensor(f"{prefix}.mlp.gate_proj.weight", INTERMEDIATE, HIDDEN, scale=0.05),
            tensor(f"{prefix}.mlp.up_proj.weight", INTERMEDIATE, HIDDEN, scale=0.05),
            tensor(f"{prefix}.mlp.down_proj.weight", HIDDEN, INTERMEDIATE, scale=0.05),
        ]
    write_safetensors(out / "model.safetensors", tensors)
    print(out)


if __name__ == "__main__":
    main()
