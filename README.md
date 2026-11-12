<!-- wisent-banner:start -->
<p align="center">
  <img src="assets/readme-banner.webp" alt="ster by Wisent" width="100%">
</p>
<!-- wisent-banner:end -->

<!-- wisent-readme-signals:start -->
[![Source](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/wisent-ai/ster) [![Issues](https://img.shields.io/badge/GitHub-Issues-181717?logo=github)](https://github.com/wisent-ai/ster/issues) [![Wisent](https://img.shields.io/badge/Wisent-Website-0B0B0B)](https://wisent.com) [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/wisent-ai/) [![X](https://img.shields.io/badge/X-Follow-000000?logo=x&logoColor=white)](https://x.com/wisentai) [![Enterprise](https://img.shields.io/badge/Enterprise-Book%20a%20call-0B0B0B?logo=calendly)](https://calendly.com/lbartoszcze)
<!-- wisent-readme-signals:end -->

[![Source](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/wisent-ai/ster)
[![License](https://img.shields.io/github/license/wisent-ai/ster)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54)

# Ster

Ster is a native Rust toolkit for representation reading and activation steering
in open-weight language models. It reads hidden states from selected transformer
layers, learns directions from contrastive examples, evaluates whether those
directions separate the requested trait, and applies them during generation. It
also trains the weights themselves: LoRA adapters under a supervised, a
preference, a reward-modelling or a policy-gradient objective, and the tools to
merge, score and inspect what comes out. And it reads decisions: typed
questions about a state answered from one forward pass, with a probability for
every option and no text generated.

The product is **Ster**. Wisent is the company that builds it.

## Current product contract

Ster 0.14 provides one binary and one library crate. Both use the same versioned
JSON artifacts and native Candle runtime.

Included now:

- local and Hugging Face checkpoints published as unquantized Safetensors
  whose `model_type` is `llama`, `mistral`, `mixtral`, `qwen2`, `qwen2_moe`,
  `qwen3`, `qwen3_moe`, `phi`, `phi3`, `granite`, `granitemoe`, `stablelm`,
  `starcoder2`, `cohere`, `cohere2`, `nemotron`, `olmo`, `olmo2`, `olmo3`,
  `olmoe`, `exaone4`, `internlm3`, `seed_oss`, `arcee`, `ernie4_5`,
  `minicpm`, `orion`, `glm`, `glm4`, `gpt_neox`, `gptj`, `gpt2`,
  `gpt_bigcode`, `opt`, `bloom`, `falcon`, `mpt`, `deepseek_v2`,
  `deepseek_v3` (also inside `kimi_k25`), `minicpm3`, `mamba`, `falcon_mamba`, `glm4_moe`,
  `internlm2`, `exaone`, `jamba`, `hunyuan_v1_dense`, `mamba2`, `bamba`,
  `gpt_oss`, `lfm2`, `ernie4_5_moe`, `dbrx`, `phimoe`, `hunyuan_v1_moe`,
  `telechat`, `lfm2_moe`, `granitemoehybrid`, `nemotron_h`, `jais2`,
  `bailing_moe`, `bailing_hybrid`, `sarvam_moe`, `sarvam_mla`, `cohere2_moe`, `deepseek`, `ministral3` (also inside `mistral3`), `llama4_text` (also inside `llama4`), `qwen3_5_text` and `qwen3_5_moe_text` (also inside `qwen3_5` and `qwen3_5_moe`), `afmoe`, `nemotron_h_puzzle`, `chatglm`, `laguna`, `longcat_flash_ngram` (also a config that names only `LongcatFlashNgramForCausalLM`), `muse_glimmer_text` (also inside `muse_glimmer`), `plamo3`, `hrm_text`, `inkling_text` (also inside `inkling_mm_model`), `deepseek_v4`, `glm5_next_text` (also inside `glm5_next`), `hy_v4` (in its release tensor names or Transformers'), `dots3_note` (sparse-indexed latent attention on its `full_attention` layers, its own narrower latent attention through `sliding_window_size` on its `sliding_attention` layers, both with `k_rope_only_layernorm`, `apply_mla_qkv_lora_rescale` and a headwise `g_proj` gate; the image and audio towers are never mapped), `minimax_m3_vl_text` (inside `minimax_m3_vl` below `language_model`, or a config that names only `MiniMaxM3SparseForCausalLM`: every feed-forward is SwiGLU-OAI clamped by `swiglu_limit` with the gate sharpness `swiglu_alpha`, every norm is Gemma's, and the `sparse_attention_freq` layers run MiniMax Sparse Attention, each key-value group attending only the `sparse_topk_blocks` blocks of `sparse_block_size` keys its block indexer ranks highest, its own and `sparse_local_block` local blocks among them; the vision tower is never mapped), `TeleFLM`, `falcon_h1`, `qwen3_next`, `kimi_linear` (also inside `kimi_k3`), `minimax_m2`, `minimax_text_01` (or `minimax`), `zamba2`, `step3_text` (also inside `step3_vl`), `step3p5`, `k2_horizon`, `longcat_flash`, `mimo`, `mimo_v2_flash`, `mimo_v2`, `mellum`, `flex_olmo`, `granite_swa`, `granitemoe_swa`, `granitemoeshared`, `glm4_moe_lite`, `glm_moe_dsa`, `deepseek_v32`, `axk1`, `iquestcoder`, `iquestloopcoder`, `hyperclovax` (also inside `hyperclovax_vision_v2`), `apertus`, `exaone_moe`, `solar`, `step1`, `telechat3`, `param2moe`, `PanguEmbedded`, `pangu_ultra_moe`, `olmo_hybrid`, `hy_v3`, `nanbeige`, `nemotron-nas`, `smollm3`, `gemma`, `gemma2`, `vaultgemma`, `gemma3_text`, `gemma3n_text` (also inside `gemma3n`), `gemma4_text` or `gemma4_unified_text` (also inside `gemma4` and `gemma4_unified`), with the
  rotation read from
  `rope_theta` and `rope_scaling` or from Transformers 5's `rope_parameters`.
  OLMo, OLMoE and DBRX clamp
  every query, key and
  value component to `clip_qkv`; PhiMoE routes two experts per token by
  SparseMixer. GPT-OSS
  adds a learned sink logit per head to every attention softmax and routes
  over biased experts with a clamped gate (`swiglu_limit`); OpenAI's own
  checkpoints are MXFP4-quantized and refused, their BF16 exports load.
  LFM2 replaces attention on most layers with a gated short convolution
  (`conv.in_proj` to `B`, `C`, `x`; a causal depthwise `conv` over `B·x`
  spanning `conv_L_cache` positions; `C` times its output; `conv.out_proj`)
  and keeps attention with per-head norms on `full_attn_idxs`; adapter
  training on it is refused.
  Coverage is measured
  against vLLM's list of text-only architectures; every architecture on it
  that Ster does not load is tracked with what the decoder lacks for it.
  Mamba and Falcon-Mamba replace attention with a selective state-space
  mixer (`in_proj`, a causal depthwise `conv1d`, `x_proj`, `dt_proj`,
  `A_log`, `D`, `out_proj`, scanned token by token and carried between
  decode steps); Jamba interleaves that mixer, with weighted norms on its
  step and matrices, with position-free attention, and gives every layer a
  feed-forward, routed on its expert layers. Mamba-2 uses the structured
  multi-head mixer (`in_proj` to gate, stream, `B`, `C` and a step per head,
  `conv1d`, `dt_bias`, `A_log`, `D`, a gated RMS `norm` per group,
  `out_proj`), and Bamba interleaves it with attention on
  `attn_layer_indices` and a feed-forward after every layer. Configs written
  with Python's `Infinity`, `-Infinity` or `NaN` are read with those as
  unstated. These state-space models are steered like any
  other model, and adapter training on them is refused because their
  state-space layers have no attention or feed-forward projection. Gemma 3's
  image-text checkpoints (`model_type` `gemma3`) load as their `text_config`
  language model, read from below `language_model`; the vision tower is
  never mapped. Every
  other family is a rotary, grouped-query decoder with what its config
  adds read from the config itself: a stated `head_dim` (attention wider or
  narrower than the residual stream), RMS or LayerNorm norms (with a bias
  for StableLM, Starcoder2, Phi-2, Nemotron and Orion, without for Cohere,
  with neither scale nor bias for OLMo 1, offset from one for Gemma and
  Nemotron), query and key norms per head (Qwen3,
  Gemma 3, Cohere's `use_qk_norm`, StableLM's `qk_layernorm`) or over the
  whole projection (OLMo 2 and 3, OLMoE), projection bias (`attention_bias`,
  `use_qkv_bias`, `use_bias`, `mlp_bias`, Phi-2's biased head), a gated
  feed-forward, a plain one (Starcoder2's `c_fc` and `c_proj`, Phi-2's `fc1`
  and `fc2`, Nemotron's squared ReLU) or a mixture of experts (Mixtral,
  Qwen2-MoE with its gated shared expert, Qwen3-MoE, OLMoE, GraniteMoE,
  DeepSeek with its shared experts, sigmoid scores, group-limited routing
  and `e_score_correction_bias`: a router picks `num_experts_per_tok`
  experts per token, renormalised when the family does, with Qwen's
  `mlp_only_layers` and `decoder_sparse_step` and DeepSeek's
  `first_k_dense_replace` layers kept dense), DeepSeek's and MiniCPM3's
  multi-head latent attention (low-rank query and key-value bottlenecks, a
  rotated key part shared by every head, values narrower than queries),
  sequential or parallel blocks (Cohere, Phi-2,
  StableLM's `use_parallel_residual`),
  sliding-window attention on the layers `sliding_window`,
  `max_window_layers`, `sliding_window_pattern` or `layer_types` name,
  Phi-3's fused `qkv_proj` and `gate_up_proj`, GLM's fused `gate_up_proj`,
  GPT-2's, GPT-BigCode's and MPT's stacked `c_attn`/`Wqkv` and GPT-NeoX's,
  BLOOM's and Falcon's `query_key_value` grouped by key-value head (split at
  load, and merged back row by
  row by `ster tune merge`), GPT-2's `Conv1D` weights stored inputs-first,
  positions as rotation, a learned table (GPT-2, GPT-BigCode, OPT, MPT
  without ALiBi) or ALiBi (BLOOM, MPT, Falcon's `alibi`), BLOOM's norm after
  the embedding, Falcon's new decoder architecture with `ln_attn` and
  `ln_mlp`, each family's own tensor paths
  (GPT-NeoX's `gpt_neox.layers` and `embed_out`, GPT-J's and GPT-2's
  `transformer.h`, OPT's `model.decoder.layers`, checkpoints saved without
  that root), parallel blocks with one norm or two, OLMo 2's
  and EXAONE 4's norms after each sublayer instead of before it, GLM-4's
  `post_self_attn_layernorm` and `post_mlp_layernorm`, Granite's and
  MiniCPM's embedding, residual and logit multipliers, Granite's attention
  multiplier, Cohere's `logit_scale`, interleaved rotation (Cohere, GLM,
  ERNIE 4.5, GPT-J), unrotated
  layers (SmolLM3's `no_rope_layers`, Cohere 2's global layers), rotation of
  only part of each head (`partial_rotary_factor`, GPT-NeoX's `rotary_pct`,
  GPT-J's `rotary_dim`), GPT-style key names (`n_embd`, `n_layer`, `n_head`,
  `n_positions`, `n_inner`, `rotary_emb_base`), `linear` rotary
  scaling and Phi-3's `longrope` (short factors within the original context,
  long ones past it), YaRN (`rope_scaling` of type `yarn`, with DeepSeek's
  `mscale_all_dim` sharpening the scores), and Gemma's `1 + weight` norms, scaled embedding, tanh
  GELU gate and tied embeddings, plus Gemma 2's and 3's post-attention and
  post-feed-forward norms, `query_pre_attn_scalar` and logit soft-capping,
  and Gemma 3's separate `rope_local_base_freq` on sliding-window layers. A
  quantized checkpoint (one that declares `quantization_config`, such as
  GPTQ) is refused with the sentence that names it, as is a config whose
  `hidden_act` (or `activation_function`) is not silu, gelu,
  gelu_pytorch_tanh, gelu_new, gelu_fast, relu or relu2, that rotates an odd
  or empty share of each head, whose `rope_scaling` is not `llama3`,
  `linear`, Phi-3's `longrope`, `yarn` or HunYuan's `dynamic` with an
  `alpha` (a fixed base of `rope_theta * alpha^(d / (d - 2))`), or that asks for a variant the decoder
  does not implement (GPT-2's `scale_attn_by_inverse_layer_idx`, OPT's
  post-norm `do_layer_norm_before: false` or `word_embed_proj_dim`, BLOOM's
  `apply_residual_connection_post_layernorm`). Adapter targets a family has
  no projection for (the gate of a plain feed-forward, every feed-forward
  projection of a mixture of experts, the key, value and bottlenecked query
  of latent attention, anything on a state-space model) are refused before
  any adapter is built;
- CPU execution, with compile-time Metal and CUDA backends;
- pair-set authoring and inspection for duplicates, length balance and
  diversity, with no model loaded;
- synthetic pair generation from a trait description, written either by the
  local runtime or by a hosted model Stado's inference alternatives answer,
  with every answer judged for a refusal by the model that wrote it;
- last-token hidden-state extraction from any selected transformer layer;
- contrastive activation addition (`caa`), principal-direction (`pca`), and
  logistic-probe training;
- holdout selection across method and layer, published as the scored candidate
  table the choice was made on;
- artifact evaluation by pair-ordering accuracy and projection margin;
- additive residual-stream steering during autoregressive generation;
- LoRA supervised fine-tuning of local checkpoints from prompt and completion
  examples;
- direct preference optimization, and its IPO variant, over a contrastive pair
  set, scored against the frozen reference the same weights already carry;
- Bradley-Terry reward models: a scalar head trained with the adapters beneath
  it and written in the same artifact;
- group-relative policy optimization against a reward model or an offline
  deterministic reward, with a KL penalty to the frozen base;
- merging an adapter into the base weights as a standalone checkpoint;
- frozen adapter artifacts applied at generation time;
- deterministic JSON pair, activation, steering, and evaluation formats.
- typed decisions — a choice from named options, a score on ordered levels,
  or a yes/no — read from the next-token distribution over answer letters
  after one cached pass over the state, every option shown under every
  letter so the model's letter preference cancels, in TypeSafe's System One
  request and answer shapes;
- temperature calibration of those decisions on labelled examples, reported
  with negative log-likelihood, expected calibration error, accuracy, and a
  shuffled-state control;

Explicit boundaries:

- The native runtime implements the decoder families above and no others.
  A config naming any other `model_type` — an encoder-decoder such as T5
  among them, since every family above is one decoder stack — fails before
  weights are loaded
  rather than silently running a wrong decoder.
- Ster controls local open-weight models. Hosted model routing belongs to
  Stado's inference alternatives (Brama among them).
- Steering reads hidden states, so it always runs on a local open-weight model.
  Writing pair text needs no activations, so `ster pairs synthesize` may take
  its generator from `stado capabilities call inference --consumer ster`
  instead. Ster holds no provider credential, names no gateway and speaks no
  provider API.
- Fine-tuning trains low-rank adapters, and on a reward run the scalar head
  that reads them. It trains nothing else: the base weights are mapped
  read-only and never registered as trainable. One sequence goes through each
  forward pass with gradient accumulation standing in for a batch, and there is
  no distributed training and no fleet placement. Ster does not place work on
  another machine, and nothing places Ster: it trains on the machine you start
  it on.
- No objective consults a hosted model. `ster tune grpo` takes its reward from
  a reward model you trained or from a deterministic function of the
  completion; there is no judge model and no LLM-as-critic wired into any
  gradient.
- Ster does not place work on a fleet, and manages no credentials: it has no
  credential store, no credential lifecycle, and reads what it needs from the
  environment it was started in. Neither responsibility is delegated to another
  product. The compute registry declares no Ster placement profile, no Ster OS
  unit, and no Ster workload; `stado fleet list` places Ster nowhere; and
  Skarbiec has no record of Ster in its source, README or documentation. If you
  need a run on a bigger machine, you start Ster on that machine yourself.
- Release delivery is the one thing another product does do for Ster: Stado
  installs the built binary from the `.wisent-release.json` this repository
  ships. See [Installing through Stado](#installing-through-stado).
- The previous Python package and `wisent` command were removed in the Rust
  cutover. Python namespace compatibility is not part of the Ster contract.

## Install

Install the current source release from GitHub:

```bash
cargo install --git https://github.com/wisent-ai/ster --locked
```

From this source checkout:

```bash
cargo install --path . --locked
```

Metal and CUDA are build-time choices:

```bash
cargo install --git https://github.com/wisent-ai/ster --features metal --locked
cargo install --git https://github.com/wisent-ai/ster --features cuda --locked
```

The crates.io name `ster` is currently unclaimed and is not Ster's release
surface. `pip install ster` installs unrelated software from another publisher.

Delivery through Stado and the checkpoint cache are covered in
[installing and where downloads land](docs/guide/install.md).


## First steering workflow

From a pair set to a steered generation — importing or writing the set,
adding, synthesizing, importing and inspecting pairs, training a direction and
generating with it — is walked through in the
[first steering workflow](docs/guide/start/first-workflow.md).

## CLI

Every command, what the Python `wisent` CLI called it, and the guide of each
command family are listed in the [CLI overview](docs/guide/start/cli.md). Run
`ster <command> --help` for exact arguments; every answer is one JSON document
on stdout, and the global `--text` flag prints it as `path: value` lines.

Ster reports nothing until you answer. The first command you run on a terminal
asks once whether it may send its maintainers its failures (`errors`), its
failures and the commands that ran (`usage`), or nothing; a script or a coding
agent is never asked. No file, prompt, pair or command argument is sent, and
your home directory and host name are taken out first. `ster telemetry
status|on|off|show|export|flush` reads and changes it and shows exactly what
waits to be sent, and `ster telemetry events` lists every kind of event Ster
can send, the consent each needs and every property it carries
([reference](https://ster.wisent.com/docs/cli/telemetry)); `DO_NOT_TRACK=1`
sends nothing.

`ster support report --description "what you did and what happened"
[--subject …] [--email …] [--without-failure]` asks Ster's support for help,
with Ster's version, the system's, this installation's id and the last failure
telemetry holds; `ster support status` prints every report this installation
filed with its status and replies
([reference](https://ster.wisent.com/docs/cli/support)). Ster Desktop's Help →
Report a Problem files and reads the same reports.

`ster doctor [--json]` tries every functionality Ster's failure catalogue
declares with an entry point, now, on this machine: the telemetry directory
can be written and where events go, where support reports go and their
replies, the workspace and its active pair set. Each is said to work with what
was seen, to fail with what failed and how to fix it, or to be skipped because
it changes your data; it exits 1 when one failed
([reference](https://ster.wisent.com/docs/cli/doctor)).

## Architecture

The runtime uses Candle directly. Ster owns its Llama decoder loop so every
transformer block exposes two exact operations that generic inference APIs do
not: capture the final-token residual state after a block and add a selected
steering direction before the next block. The same decoder can also run a
differentiable pass, which is what makes fine-tuning possible at all: Candle's
fused `rotary_emb::rope`, `ops::softmax_last_dim`, and `ops::rms_norm` kernels
have no backward pass, so training selects composed equivalents at exactly those
three call sites while inference keeps the fused ones. The same pass also
chooses whether the attached adapters apply, which is what lets preference
optimization score the frozen reference without a second copy of the weights.
Tokenization, Safetensors loading, attention, KV caching, sampling, and device
kernels remain native Rust.

## Documentation and support

- Product documentation: https://ster.wisent.com/docs
- Source and defects: https://github.com/wisent-ai/ster
- Community: https://discord.gg/qRjpkthq54
- Private vulnerabilities: GitHub Security Advisories for this repository

Ster is pre-1.0. Artifact schema changes and supported-model expansion remain
subject to the repository's versioned release contract.

## License

MIT — see [LICENSE](LICENSE).
