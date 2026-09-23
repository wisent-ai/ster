//! Running the pair-authoring operations: workspace import, audit, save, and
//! synthesis with either a local runtime or a hosted writer.

use anyhow::{bail, Result};
use serde_json::{json, Value};

use std::path::Path;

use crate::{
    brama,
    pairs::{self, quality::dedupe::DedupeOptions, InspectOptions, SynthesisOptions},
    pairs::quality::diversity::DEFAULT_MAX_SAMPLE,
    ChatChoice, ContrastivePair, GenerationOptions, PairSet,

};

use super::super::requests::{
    PairsInspectRequest, PairsSaveRequest, PairsSynthesizeRequest, WorkspaceImportPairsRequest,
};

pub(in crate::request) fn workspace_import_pairs_job(request: WorkspaceImportPairsRequest) -> Result<Value> {
    let report = crate::workspace::import_pair_set(
        Path::new(&request.source),
        request.name.as_deref(),
    )?;
    Ok(serde_json::to_value(report)?)
}

pub(in crate::request) fn pairs_inspect_job(request: PairsInspectRequest) -> Result<Value> {
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let options = InspectOptions {
        dedupe: DedupeOptions {
            threshold_bits: request.dedupe_bits,
            num_bands: request.dedupe_bands,
            ..DedupeOptions::default()
        },
        refusal_threshold: request.refusal_threshold,
        ..InspectOptions::default()
    };
    let report = pairs::inspect(&pair_set, &options)?;
    Ok(serde_json::to_value(&report)?)
}

/// The editor's write path. `PairSet::save` validates before it writes, so a
/// set the loader would reject never reaches disk and the desktop sees the
/// same refusal sentence the CLI prints.
pub(in crate::request) fn pairs_save_job(request: PairsSaveRequest) -> Result<Value> {
    let pair_set = PairSet {
        trait_name: request.trait_name,
        pairs: request
            .entries
            .into_iter()
            .map(|entry| ContrastivePair { positive: entry.positive, negative: entry.negative })
            .collect(),
    };
    pair_set.save(Path::new(&request.path))?;
    Ok(json!({"path": request.path, "pairCount": pair_set.pairs.len()}))
}

pub(in crate::request) fn pairs_synthesize_job(request: PairsSynthesizeRequest) -> Result<Value> {
    let options = SynthesisOptions {
        trait_description: request.trait_description,
        trait_name: request.trait_name,
        opposite: request.opposite,
        count: request.count,
        retry_multiplier: request.retry_multiplier,
        dedupe: DedupeOptions {
            threshold_bits: request.dedupe_bits,
            num_bands: request.dedupe_bands,
            ..DedupeOptions::default()
        },
        refusal_threshold: request.refusal_threshold,
        generation: GenerationOptions {
            strength: 1.0,
            max_new_tokens: request.max_new_tokens,
            temperature: request.temperature,
            top_p: Some(request.top_p),
            seed: request.seed,
        },
        diversity_seed: request.seed,
        diversity_max_sample: DEFAULT_MAX_SAMPLE,
    };
    // Same two arms as the CLI, calling the same `pairs::synthesize`: a brama
    // request loads no weights and never touches a device.
    let (pair_set, report) = match request.generator.as_str() {
        "local" => {
            let mut runtime = request.model.load_runtime_at(&request.precision)?;
            // Synthesis is the first step of the funnel and everything
            // downstream inherits what it writes. Addressed without its
            // markers, an instruct checkpoint answers a pair request with
            // instructions about answering pair requests.
            runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
            pairs::synthesize(pairs::Generator::Local(&runtime), &options)?
        }
        "brama" => {
            // `Validate` has already refused a brama request without a route.
            let route = request.generator_model.as_deref().unwrap_or_default();
            let gateway = brama::Gateway::from_env(route)?;
            pairs::synthesize(pairs::Generator::Gateway(&gateway), &options)?
        }
        value => bail!("unknown generator {value:?}; expected local or brama"),
    };
    pair_set.save(Path::new(&request.output))?;
    Ok(json!({"path": request.output, "report": report}))
}
