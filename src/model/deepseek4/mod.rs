//! DeepSeek-V4 (`modeling_deepseek_v4.py`): manifold-constrained
//! hyper-connections around every sublayer, and the compressed attention in
//! [`attention`].
//!
//! The residual is `hc_mult` streams. Before a sublayer, a hyper-connection
//! reads all of them — flattened, RMS-normed without a scale, projected by
//! `fn` — into three weights: `pre` collapses the streams into the
//! sublayer's input, `post` spreads its output back over the streams, and
//! `comb`, made doubly stochastic by Sinkhorn-Knopp, mixes the incoming
//! streams into the outgoing ones: `out[k] = post[k] · y + Σ_j comb[j, k] ·
//! in[j]`. The model's last streams are collapsed by `hc_head` the same way
//! `pre` collapses them.
//!
//! HY-V4's independent hyper-connections (`HYV4HyperConnection`) read only
//! `pre` and `post` from the same projection and keep each incoming stream
//! as it is: `out[k] = post[k] · y + in[k]`, with `post` scaled by
//! `hc_magnitude` and lifted by `hc_eps`.

pub(super) mod attention;
mod compressor;
pub(crate) mod names;

pub(super) use compressor::CompressorState;

use candle_core::{D, DType, Tensor};
use candle_nn::VarBuilder;

use super::HyperForm;

/// The unscaled RMS norm DeepSeek-V4 reads its streams through.
fn unscaled_rms(input: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let variance = input.sqr()?.mean_keepdim(D::Minus1)?;
    input.broadcast_div(&(variance + eps)?.sqrt()?)
}

fn sigmoid(input: &Tensor) -> candle_core::Result<Tensor> {
    (input.neg()?.exp()? + 1.0)?.recip()
}

/// One sublayer's hyper-connection (`attn_hc` or `ffn_hc`).
#[derive(Debug, Clone)]
pub(super) struct HyperConnection {
    /// `fn`, `[(2 + N) · N, N · hidden]` for the manifold form and
    /// `[2 · N, N · hidden]` for the independent one, `base` and the `scale`s
    /// (three and two), in F32.
    projection: Tensor,
    base: Tensor,
    scales: Vec<f64>,
    streams: usize,
    form: HyperForm,
    eps: f64,
    norm_eps: f64,
}

/// What a hyper-connection makes of the streams: `post` `[batch, sequence,
/// N]`, `comb` `[batch, sequence, N, N]` (none for the independent form,
/// which keeps every stream) and the collapsed input `[batch, sequence,
/// hidden]`.
pub(super) struct Mixing {
    pub post: Tensor,
    pub comb: Option<Tensor>,
    pub collapsed: Tensor,
}

impl HyperConnection {
    /// `builder` is the layer's and `kind` `attn` or `ffn`: Transformers
    /// names the parameters `{kind}_hc.fn`, `.base` and `.scale`, DeepSeek's
    /// and GLM's own checkpoints `hc_{kind}_fn`, `_base` and `_scale`.
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        kind: &str,
        hidden: usize,
        streams: usize,
        form: HyperForm,
        eps: f64,
        norm_eps: f64,
    ) -> candle_core::Result<Self> {
        // The manifold form reads `pre`, `post` and an N × N `comb`, each
        // with its own scale; the independent form `pre` and `post` only.
        let (width, scale_count) = match form {
            HyperForm::Manifold { .. } => ((2 + streams) * streams, 3),
            HyperForm::Independent { .. } => (2 * streams, 2),
        };
        let module = format!("{kind}_hc");
        let transformers = builder.contains_tensor(&format!("{module}.fn"));
        let name = |part: &str| -> String {
            if transformers {
                format!("{module}.{part}")
            } else {
                format!("hc_{kind}_{part}")
            }
        };
        let scales = builder
            .get(scale_count, &name("scale"))?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect();
        Ok(Self {
            projection: builder
                .get((width, streams * hidden), &name("fn"))?
                .to_dtype(DType::F32)?,
            base: builder.get(width, &name("base"))?.to_dtype(DType::F32)?,
            scales,
            streams,
            form,
            eps,
            norm_eps,
        })
    }

    /// `streams` is `[batch, sequence, N, hidden]`.
    pub(super) fn mixing(&self, streams: &Tensor) -> candle_core::Result<Mixing> {
        let (batch, sequence, count, hidden) = streams.dims4()?;
        let n = self.streams;
        let flat = streams
            .reshape((batch, sequence, count * hidden))?
            .to_dtype(DType::F32)?;
        let mixes = unscaled_rms(&flat, self.norm_eps)?.broadcast_matmul(&self.projection.t()?)?;
        let part = |start: usize, width: usize| -> candle_core::Result<(Tensor, Tensor)> {
            Ok((
                mixes.narrow(D::Minus1, start, width)?,
                self.base.narrow(0, start, width)?,
            ))
        };
        let (pre, pre_base) = part(0, n)?;
        let (post, post_base) = part(n, n)?;
        let pre = (sigmoid(&(pre * self.scales[0])?.broadcast_add(&pre_base)?)? + self.eps)?;
        let gate = sigmoid(&(post * self.scales[1])?.broadcast_add(&post_base)?)?;
        let collapsed = collapse(streams, &pre)?;
        let iterations = match self.form {
            HyperForm::Manifold {
                sinkhorn_iterations,
            } => sinkhorn_iterations,
            HyperForm::Independent { magnitude } => {
                return Ok(Mixing {
                    post: ((gate * magnitude)? + self.eps)?,
                    comb: None,
                    collapsed,
                });
            }
        };
        let post = (gate * 2.0)?;
        let (comb, comb_base) = part(2 * n, n * n)?;
        let comb = (comb * self.scales[2])?
            .broadcast_add(&comb_base)?
            .reshape((batch, sequence, n, n))?;
        let mut comb = (candle_nn::ops::softmax(&comb, D::Minus1)? + self.eps)?;
        comb = comb.broadcast_div(&(comb.sum_keepdim(2)? + self.eps)?)?;
        for _ in 1..iterations {
            comb = comb.broadcast_div(&(comb.sum_keepdim(3)? + self.eps)?)?;
            comb = comb.broadcast_div(&(comb.sum_keepdim(2)? + self.eps)?)?;
        }
        Ok(Mixing {
            post,
            comb: Some(comb),
            collapsed,
        })
    }
}

impl Mixing {
    /// The outgoing streams: `post` spreading `output` `[batch, sequence,
    /// hidden]` plus `comb`'s mix of the incoming `streams`, or the incoming
    /// streams as they are.
    pub(super) fn spread(&self, output: &Tensor, streams: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = streams.dtype();
        let spread = self
            .post
            .to_dtype(dtype)?
            .unsqueeze(3)?
            .broadcast_mul(&output.unsqueeze(2)?)?;
        match &self.comb {
            Some(comb) => {
                let mixed = comb
                    .to_dtype(dtype)?
                    .transpose(2, 3)?
                    .contiguous()?
                    .matmul(&streams.contiguous()?)?;
                spread + mixed
            }
            None => spread + streams,
        }
    }
}

/// `Σ_n weights[n] · streams[n]`, `weights` `[batch, sequence, N]` in F32.
fn collapse(streams: &Tensor, weights: &Tensor) -> candle_core::Result<Tensor> {
    let dtype = streams.dtype();
    streams
        .to_dtype(DType::F32)?
        .broadcast_mul(&weights.unsqueeze(3)?)?
        .sum(2)?
        .to_dtype(dtype)
}

/// The model's last collapse of its streams (`hc_head`).
#[derive(Debug, Clone)]
pub(super) struct HyperHead {
    projection: Tensor,
    base: Tensor,
    scale: f64,
    eps: f64,
    norm_eps: f64,
}

impl HyperHead {
    /// `builder` is the model root's.
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        streams: usize,
        eps: f64,
        norm_eps: f64,
    ) -> candle_core::Result<Self> {
        let head = builder.pp("hc_head");
        let scale = head
            .get(1, "hc_scale")?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?[0];
        Ok(Self {
            projection: head
                .get((streams, streams * hidden), "hc_fn")?
                .to_dtype(DType::F32)?,
            base: head.get(streams, "hc_base")?.to_dtype(DType::F32)?,
            scale: f64::from(scale),
            eps,
            norm_eps,
        })
    }

    pub(super) fn collapse(&self, streams: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, sequence, count, hidden) = streams.dims4()?;
        let flat = streams
            .reshape((batch, sequence, count * hidden))?
            .to_dtype(DType::F32)?;
        let mixes = unscaled_rms(&flat, self.norm_eps)?.broadcast_matmul(&self.projection.t()?)?;
        let pre = (sigmoid(&(mixes * self.scale)?.broadcast_add(&self.base)?)? + self.eps)?;
        collapse(streams, &pre)
    }
}
