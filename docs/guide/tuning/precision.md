# Precision

The dtype the frozen base weights are mapped at, what stays in f32 whatever
that says, and why a score is only comparable with another score taken at the
same precision.



`--precision` names the dtype the frozen base weights are mapped at — `f32`,
`f16` or `bf16` — and defaults to `f32`, so every run recorded before this flag
existed is unchanged. It is on every command that maps a checkpoint — `train`,
`optimize`, `evaluate`, `generate`, `extract`, `pairs synthesize`, and the five
`ster tune` subcommands that load a model — and mirrored as `precision` on each
matching `/v1` endpoint. `tune merge` and `tune inspect` take none: neither
builds a decoder.

It names the base weights and nothing else. Adapters, a reward run's scalar
head, and every `AdamW` moment stay in F32 whatever it says, and that split is
the whole of mixed precision rather than a detail of it: a low-rank update is
small relative to the weight it corrects, and an update below that weight's own
ulp rounds to nothing in half precision, so the adapter would train while the
model did not move. Candle makes the split nearly free — `AdamW` builds both
moments at each variable's own dtype, `lora::Adapter::forward` already casts
each factor to the activation's dtype, and `to_dtype` is differentiable with a
backward that casts the gradient back — so an F32 adapter stays F32 through a
half-precision forward without anything being arranged.

Three things in the forward pass are held at F32 regardless, because they are
the places where half precision is wrong rather than merely cheaper: attention
promotes queries, keys and values before the score matmul and casts only the
output back; the rotary angles and the rotation itself run in F32, because a
position is an absolute index rather than a weight and a half mantissa there
makes neighbouring late positions round to the same angle — a phase error that
reads as a slightly different sentence and never as a numerical fault; and every
loss is summed in F32, because a log-softmax adds tens of thousands of terms
into one accumulator and in half the smallest of them stop changing it. The
key-value cache and the residual stream still store half-width values, so the
memory saving survives all three.

The rotary one has a number behind it. Scoring two held-out examples of about
1860 tokens on TinyLlama-1.1B-Chat, against the same model's own `f32` loss of
1.795495907465617: rotating in F32 costs 0.07% and rotating in half costs 1.1%,
so the promotion removes roughly fifteen sixteenths of what half precision
would otherwise take. At 128 tokens the two are indistinguishable — no position
is far enough out for a half mantissa to collide two angles — which is the
honest limit of the short fixtures and the reason the long run was worth doing.

A steering vector is cast to the model's dtype on the way in, and it survives
the cast as the same direction. Each vector in an artifact is unit-normalized
F32; at `f16` the mean relative error per component is 1.8e-4, the largest
absolute error 2.9e-5, and the cosine similarity with the original is 1.0 to
within F32's own rounding. Two to four components in 2048 fall below half
precision's smallest normal and become subnormal; none flush to zero, and they
are the near-zero components that carry none of the direction. Generating from
TinyLlama-1.1B-Chat with the same artifact, strength, seed and greedy sampling,
`f32` and `f16` produce character-identical text at strengths 1 and 2 over 32
and 96 tokens on two prompts, steered and unsteered. At strength 4 both collapse
to an immediate end-of-sequence, which is over-steering rather than a precision
effect.

`bf16` on the `cpu` device is refused, at load, before a weight is mapped:

```text
bf16 has no CPU matmul kernel in this Candle build; use --precision f16 for half precision on cpu, or --device metal for bf16
```

That is a fact about Candle rather than a policy. `cpu_backend`'s matmul accepts
F16, F32 and F64 and returns `unsupported dtype BF16 for op matmul` for anything
else, so a bf16 CPU run does not run slowly — it downloads and maps the whole
checkpoint and then dies at the first projection. `f16` on the CPU is real half
arithmetic on an Apple-silicon class machine: `gemm` selects a native
`neonfp16` microkernel on aarch64 when the hardware reports the `fp16` feature.
On `metal` all three work.

Measured on `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, CPU, revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`, scoring the eight checked-in
examples with `ster tune evaluate --max-sequence 128`:

| | peak resident | user CPU | loss | perplexity |
| --- | --- | --- | --- | --- |
| `--precision f32` | 6.46 GiB | 15.2 s | 4.18851804357814 | 65.92502052501293 |
| `--precision f16` | 4.29 GiB | 10.2 s | 4.17449491605984 | 65.00699737728682 |

Two runs of each, peak resident stable to 0.02% and user CPU to 4%; wall clock
is not reported because the machine was shared and the same run varied between
3.9 s and 10.8 s while its CPU time did not move. The 2.32 GiB saved is the
checkpoint's own weight count at two bytes instead of four. The loss differs by
0.33%, which is what half precision cost in accuracy here, and each precision
reproduces itself exactly.

Training is where it decides whether a run happens at all. The same command as
an SFT run over all 22 layers —
`ster tune sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --examples docs/examples/tuning/examples.json --epochs 1 --max-sequence 128 --seed 42`
— trains 88 adapter tensors and 1126400 parameters, and asks for a peak memory
footprint of 102.8 GB at `f32` against 55.6 GB at `f16`, a factor of 1.85. On a
quiet machine both finish; on a loaded one the `f32` run was killed by the
operating system partway through the first epoch — not a Ster refusal and not a
Candle allocation failure, just a signal — while `f16` completed at a final loss
of 4.200512409210205. The same `f32` command completed on a second, quieter
machine, at 76.9 s of user time against 105.6 s of system time and 2.4 million
involuntary context switches, which is the shape of a box compressing memory to
stay alive.

Footprint is quoted rather than peak resident for the training runs, because
resident moved between 23.9 and 34.6 GiB across machines while the footprint
held: resident reflects what the operating system let the process keep, and the
footprint reflects what it asked for. And most of that footprint is the
autograd tape over 22 differentiable layers rather than the weights. The tape is
not what `--precision` controls; the way to shrink it is fewer differentiable
layers, not a narrower dtype. What halving the base weights buys is the margin
that decides whether a loaded machine finishes the run.

