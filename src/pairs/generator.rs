//! Which model writes the pair text, and the one call each candidate costs.

use anyhow::Result;

use crate::runtime::{GenerationOptions, Runtime};

use super::synthesis::SynthesisOptions;

/// Where the pair text comes from. Steering always stays local; only the
/// writer of the training data may be hosted, because a pair is plain text
/// and no hidden state is read to produce it.
#[derive(Clone, Copy)]
pub enum Generator<'a> {
    Local(&'a Runtime),
    Gateway(&'a crate::brama::Gateway),
}

impl Generator<'_> {
    /// Names the generator in reports and progress lines.
    pub fn label(&self) -> String {
        match self {
            Self::Local(runtime) => format!("local:{}", runtime.model_id),
            Self::Gateway(gateway) => format!("brama:{}", gateway.model()),
        }
    }

    /// One call to whichever model is writing, with `call` counting the calls
    /// already made in this run.
    ///
    /// Local: `Runtime::generate` builds a fresh `LogitsProcessor` from
    /// `GenerationOptions::seed` on every call, so a fixed seed would replay
    /// the identical continuation for the identical prompt and the run would
    /// collapse to a single deduplicated pair. Advancing the seed by the call
    /// index makes each call an independent draw while keeping the whole run
    /// reproducible from the one seed the caller supplied.
    ///
    /// Gateway: neither the seed nor `top_p` travels. Brama's chat request
    /// carries `max_tokens` and `temperature` and has no field for either, and
    /// the provider behind the route owns its own sampler — so a hosted run is
    /// not reproducible from `--seed`, and the running dedupe is what keeps
    /// repeated draws out of the set.
    pub(super) fn generate(
        &self,
        prompt: &str,
        options: &SynthesisOptions,
        call: u64,
    ) -> Result<String> {
        let text = match self {
            Self::Local(runtime) => {
                let generation = GenerationOptions {
                    seed: options.generation.seed.wrapping_add(call),
                    ..options.generation
                };
                runtime.generate(prompt, None, generation)?
            }
            Self::Gateway(gateway) => gateway.complete(
                prompt,
                options.generation.max_new_tokens,
                options.generation.temperature,
            )?,
        };
        Ok(text.trim().to_owned())
    }
}
