# Architecture

[Ster](../../../README.md) · [CLI overview](cli.md)

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
