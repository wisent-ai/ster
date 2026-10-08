//! Running the pair-authoring operations: workspace import, audit, save, and
//! synthesis with either a local runtime or a hosted writer.

use anyhow::{Result, bail};
use serde_json::{Value, json};

use std::path::Path;

use crate::{
    ChatChoice, ContrastivePair, GenerationOptions, PairSet, brama,
    pairs::{self, InspectOptions, SynthesisOptions, quality::dedupe::DedupeOptions},
};

use super::super::requests::{
    PairsImportRequest, PairsInspectRequest, PairsMergeRequest, PairsSaveRequest,
    PairsSynthesizeRequest, WorkspaceImportPairsRequest, WorkspacePairSetRequest,
    WorkspaceShowRequest,
};

/// The same merge as `ster pairs merge`, and the same document.
pub(in crate::request) fn pairs_merge_job(request: PairsMergeRequest) -> Result<Value> {
    let sources: Vec<std::path::PathBuf> = request.sources.iter().map(std::path::PathBuf::from).collect();
    let (set, report) = pairs::merge(&sources, &request.trait_name)?;
    set.save(Path::new(&request.output))?;
    Ok(json!({"output": request.output, "report": report}))
}

pub(in crate::request) fn workspace_import_pairs_job(
    request: WorkspaceImportPairsRequest,
) -> Result<Value> {
    let report =
        crate::workspace::import_pair_set(Path::new(&request.source), request.name.as_deref())?;
    Ok(serde_json::to_value(report)?)
}

/// The same answer as `ster workspace show`.
pub(in crate::request) fn workspace_show_job(_request: WorkspaceShowRequest) -> Result<Value> {
    Ok(serde_json::to_value(crate::workspace::summary()?)?)
}

/// The same operation as `ster workspace select`.
pub(in crate::request) fn workspace_select_job(request: WorkspacePairSetRequest) -> Result<Value> {
    Ok(serde_json::to_value(crate::workspace::select_pair_set(
        &request.id,
    )?)?)
}

/// The same operation as `ster workspace remove`.
pub(in crate::request) fn workspace_remove_job(request: WorkspacePairSetRequest) -> Result<Value> {
    Ok(serde_json::to_value(crate::workspace::remove_pair_set(
        &request.id,
    )?)?)
}

/// A benchmark export read into a pair set by the same `pairs::benchmark`
/// the `ster pairs import` command runs; `report.skipped` names every row
/// that could not become a pair.
pub(in crate::request) fn pairs_import_job(request: PairsImportRequest) -> Result<Value> {
    let options = pairs::benchmark::ImportOptions {
        benchmark: pairs::benchmark::Benchmark::parse(&request.benchmark)?,
        trait_name: request
            .trait_name
            .unwrap_or_else(|| request.benchmark.clone()),
        source: request.source.into(),
        examples: request.examples.map(Into::into),
        fields: pairs::benchmark::ChoiceFields::from_parts(
            request.question,
            request.choices,
            request.answer,
            request.answer_form,
            request.labels,
        )?,
        count: request.count,
        seed: request.seed,
    };
    let (set, report) = pairs::benchmark::import(&options)?;
    set.save(Path::new(&request.output))?;
    Ok(json!({"output": request.output, "trait_name": set.trait_name, "report": report}))
}

pub(in crate::request) fn pairs_inspect_job(request: PairsInspectRequest) -> Result<Value> {
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let options = InspectOptions {
        dedupe: DedupeOptions::new(request.dedupe_bits, request.dedupe_bands),
        refusal_threshold: request.refusal_threshold,
        unbalanced_ratio: request.unbalanced_ratio,
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
            .map(|entry| ContrastivePair {
                positive: entry.positive,
                negative: entry.negative,
            })
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
        dedupe: DedupeOptions::new(request.dedupe_bits, request.dedupe_bands),
        refusal_threshold: request.refusal_threshold,
        generation: GenerationOptions {
            strength: None,
            max_new_tokens: request.max_new_tokens,
            temperature: request.temperature,
            top_p: Some(request.top_p),
            seed: request.seed,
        },
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
            runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?)?;
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
