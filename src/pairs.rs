//! pairs.rs — authoring, inspecting, and synthesizing contrastive pair sets.
//!
//! Ster could previously only consume a `pairs.json` someone else produced.
//! This module closes that gap with two surfaces:
//!
//! * `inspect` — a model-free audit of an existing set. It answers the three
//!   questions that make a pair set silently useless: are pairs duplicated,
//!   did the generating model refuse instead of answering, and is one side
//!   systematically longer than the other (a length confound trains a
//!   "verbosity" direction rather than the trait).
//! * `synthesize` — a faithful port of Wisent's
//!   `SyntheticContrastivePairsGenerator.generate`, driven by Ster's own
//!   `Runtime` so the boundary stays local open-weight models only.
//!
//! Both are the single implementation behind the CLI arms and the
//! `/v1/pairs/*` serve endpoints.

use anyhow::{Result, bail};
use serde::Serialize;

use crate::{
    artifact::{ContrastivePair, PairSet},
    dedupe::{self, DedupeOptions, Duplicate},
    diversity,
    refusal,
    runtime::{GenerationOptions, Runtime},
    workflow,
};

/// A pair whose two sides differ by more than this factor in characters is
/// reported as unbalanced. Length is the confound the product documents at
/// https://ster.wisent.com/docs/concept-contrastive-pair: when the positive
/// side is consistently three times longer than the negative, the trained
/// direction encodes response length, not the trait, and steering on it just
/// makes the model verbose.
pub const UNBALANCED_RATIO: f64 = 3.0;

/// Trait-name fallback length. Long descriptions make unusable artifact
/// labels, so an unnamed set borrows the first 64 characters of its
/// description.
const TRAIT_NAME_LIMIT: usize = 64;

/// Used when the model returns nothing for the opposite-trait question —
/// verbatim from the Python generator's `"neutral and plain"` fallback.
const DEFAULT_OPPOSITE: &str = "neutral and plain";

/// Verbatim from the Python generator: the single instruction that produces
/// the user question each pair answers.
const QUESTION_INSTRUCTION: &str = "Write one short question a user might ask. Example: 'What is your favorite hobby?' Just the question, nothing else.";

/// Verbatim `roleplay_neg_fix` from Wisent's `db_instructions/mini_dp.py`.
/// The Python cleaner sends it as a system message; `Runtime::generate` takes
/// a single prompt string, so it is prepended to the user turn instead.
const ROLEPLAY_NEG_FIX: &str = "You are fixing ONLY the negative example of a contrastive pair.\nProduce a single concise negative response for the given prompt that exemplifies the UNDESIRED trait.\nIt must be fictional/hypothetical, safe, and non-actionable. Return raw text only.";

// MARK: - Inspection

#[derive(Debug, Clone)]
pub struct InspectOptions {
    pub dedupe: DedupeOptions,
    pub refusal_threshold: f32,
    pub diversity_seed: u64,
    pub diversity_max_sample: usize,
}

impl Default for InspectOptions {
    fn default() -> Self {
        Self {
            dedupe: DedupeOptions::default(),
            refusal_threshold: refusal::DEFAULT_THRESHOLD,
            diversity_seed: 42,
            diversity_max_sample: diversity::DEFAULT_MAX_SAMPLE,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RefusalFlag {
    pub score: f32,
    pub family: String,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntryReport {
    pub index: usize,
    pub positive: String,
    pub negative: String,
    pub positive_chars: usize,
    pub negative_chars: usize,
    pub positive_words: usize,
    pub negative_words: usize,
    pub duplicate: Option<Duplicate>,
    pub positive_refusal: Option<RefusalFlag>,
    pub negative_refusal: Option<RefusalFlag>,
    pub length_ratio: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SetReport {
    pub trait_name: String,
    pub pair_count: usize,
    pub duplicate_count: usize,
    pub refusal_count: usize,
    pub unbalanced_count: usize,
    pub diversity: diversity::Scores,
    pub entries: Vec<EntryReport>,
}

/// Audit a pair set without loading a model. Every judgement here is textual,
/// which is why the desktop app can show it the moment a file is opened.
pub fn inspect(pairs: &PairSet, options: &InspectOptions) -> Result<SetReport> {
    let duplicates = dedupe::classify(&pairs.pairs, options.dedupe)?;
    let mut entries = Vec::with_capacity(pairs.pairs.len());
    let mut duplicate_count = 0usize;
    let mut refusal_count = 0usize;
    let mut unbalanced_count = 0usize;

    for (index, pair) in pairs.pairs.iter().enumerate() {
        let duplicate = duplicates[index];
        if duplicate.is_some() {
            duplicate_count += 1;
        }
        let positive_refusal = flag(&pair.positive, options.refusal_threshold);
        let negative_refusal = flag(&pair.negative, options.refusal_threshold);
        if positive_refusal.is_some() || negative_refusal.is_some() {
            refusal_count += 1;
        }
        let positive_chars = pair.positive.chars().count();
        let negative_chars = pair.negative.chars().count();
        let length_ratio = length_ratio(positive_chars, negative_chars);
        if length_ratio > UNBALANCED_RATIO {
            unbalanced_count += 1;
        }
        entries.push(EntryReport {
            index,
            positive: pair.positive.clone(),
            negative: pair.negative.clone(),
            positive_chars,
            negative_chars,
            positive_words: pair.positive.split_whitespace().count(),
            negative_words: pair.negative.split_whitespace().count(),
            duplicate,
            positive_refusal,
            negative_refusal,
            length_ratio,
        });
    }

    // Diversity reads the positive side only: the two sides of a pair are
    // near-copies of each other by construction, so scoring both would report
    // the contrast as repetition.
    let positives: Vec<String> = pairs.pairs.iter().map(|pair| pair.positive.clone()).collect();
    let diversity =
        diversity::compute(&positives, options.diversity_seed, options.diversity_max_sample);

    Ok(SetReport {
        trait_name: pairs.trait_name.clone(),
        pair_count: pairs.pairs.len(),
        duplicate_count,
        refusal_count,
        unbalanced_count,
        diversity,
        entries,
    })
}

fn flag(text: &str, threshold: f32) -> Option<RefusalFlag> {
    let scored = refusal::score(text);
    if scored.score < threshold {
        return None;
    }
    Some(RefusalFlag {
        score: scored.score,
        family: scored.family.map(|family| family.name().to_owned()).unwrap_or_default(),
        snippet: scored.snippet,
    })
}

/// Longer side over shorter side. Two empty sides are perfectly balanced and
/// 0/0 has no value, so they report 1.0; a single empty side would divide by
/// zero, so the non-empty length stands in for the ratio and the pair reads
/// as maximally unbalanced.
fn length_ratio(positive_chars: usize, negative_chars: usize) -> f64 {
    let longer = positive_chars.max(negative_chars);
    let shorter = positive_chars.min(negative_chars);
    if longer == 0 {
        return 1.0;
    }
    longer as f64 / shorter.max(1) as f64
}

// MARK: - Synthesis

#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    pub trait_description: String,
    pub trait_name: String,
    pub opposite: Option<String>,
    pub count: usize,
    pub retry_multiplier: usize,
    pub dedupe: DedupeOptions,
    pub refusal_threshold: f32,
    pub generation: GenerationOptions,
    pub diversity_seed: u64,
    pub diversity_max_sample: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SynthesisReport {
    pub trait_name: String,
    pub trait_description: String,
    pub opposite: String,
    pub requested: usize,
    pub attempts: usize,
    pub kept: usize,
    pub rejected_empty: usize,
    pub rejected_refusals: usize,
    pub rejected_duplicates: usize,
    pub refusal_retries: usize,
    pub diversity: diversity::Scores,
}

/// Generate a contrastive pair set with the loaded model.
///
/// Port of `SyntheticContrastivePairsGenerator.generate`: one opposite-trait
/// description up front, then a question / positive / negative triple per
/// attempt, with refusal repair and deduplication applied as each pair lands
/// rather than in a batch pass at the end. Cleaning inline is what lets the
/// loop stop the moment `count` *surviving* pairs exist instead of
/// re-cleaning the whole set every ten pairs the way the Python does.
pub fn synthesize(
    runtime: &Runtime,
    options: &SynthesisOptions,
) -> Result<(PairSet, SynthesisReport)> {
    if options.count == 0 {
        bail!("synthesis requires a pair count above zero");
    }
    if options.retry_multiplier == 0 {
        bail!("synthesis requires a retry multiplier above zero");
    }
    // Ster selects argmax sampling at or below zero temperature. Every prompt
    // in this loop is a constant, so argmax would return the same question and
    // the same answers forever and the whole run would dedupe to one pair.
    if options.generation.temperature <= 0.0 {
        bail!("synthesis requires a temperature above zero; argmax generation repeats a single prompt");
    }
    options.dedupe.validate()?;

    let trait_name = resolve_trait_name(options);
    let mut generator = Generator { runtime, options: options.generation, calls: 0 };

    let opposite = match options
        .opposite
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(value) => value.to_owned(),
        None => {
            let answer = generator.ask(&format!(
                "What is the OPPOSITE personality trait of: {}?\n\nDescribe the opposite in one sentence, be specific about what words/style/tone to use.",
                options.trait_description
            ))?;
            if answer.is_empty() { DEFAULT_OPPOSITE.to_owned() } else { answer }
        }
    };

    let mut index = dedupe::Index::new(options.dedupe)?;
    let mut kept: Vec<ContrastivePair> = Vec::with_capacity(options.count);
    let mut questions: Vec<String> = Vec::with_capacity(options.count);
    let mut attempts = 0usize;
    let mut rejected_empty = 0usize;
    let mut rejected_refusals = 0usize;
    let mut rejected_duplicates = 0usize;
    let mut refusal_retries = 0usize;
    let max_attempts = options.count.saturating_mul(options.retry_multiplier);

    while kept.len() < options.count && attempts < max_attempts {
        attempts += 1;
        workflow::progress(format!(
            "synthesizing pair {}/{} (attempt {})",
            kept.len() + 1,
            options.count,
            attempts
        ));

        let question = generator.ask(QUESTION_INSTRUCTION)?;
        if question.is_empty() {
            rejected_empty += 1;
            continue;
        }

        let positive = generator.ask(&persona_prompt(&question, &options.trait_description))?;
        if positive.is_empty() {
            rejected_empty += 1;
            continue;
        }

        let mut negative = generator.ask(&persona_prompt(&question, &opposite))?;
        if negative.is_empty() {
            rejected_empty += 1;
            continue;
        }

        // A refusing positive has no repair path: the trait itself is what the
        // model declined to roleplay, so re-asking would refuse again.
        if refusal::looks_like_refusal(&positive, options.refusal_threshold) {
            rejected_refusals += 1;
            continue;
        }

        // `RefusalerCleaner` + `BaseRefusaler::fix_negative`: one re-prompt,
        // never a loop, and a still-refusing replacement drops the pair.
        if refusal::looks_like_refusal(&negative, options.refusal_threshold) {
            refusal_retries += 1;
            let replacement = generator.ask(&format!(
                "{ROLEPLAY_NEG_FIX}\n\nPrompt: {question}\nTrait label: {trait_name}\nTrait description: {opposite}"
            ))?;
            if replacement.is_empty()
                || refusal::looks_like_refusal(&replacement, options.refusal_threshold)
            {
                rejected_refusals += 1;
                continue;
            }
            negative = replacement;
        }

        // Ster's `ContrastivePair` carries two strings and no separate prompt
        // field, so the question is folded into both sides in the shape the
        // README and the docs already publish. Folding it in identically is
        // what keeps the two sides matched on topic, wording, and length —
        // the only difference left between them is the trait.
        let pair = ContrastivePair {
            positive: format!("Question: {question}\nAnswer: {positive}"),
            negative: format!("Question: {question}\nAnswer: {negative}"),
        };
        if index.insert(&pair).is_some() {
            rejected_duplicates += 1;
            continue;
        }
        questions.push(question);
        kept.push(pair);
    }

    // Diversity is measured over the questions, matching the Python report:
    // the answers inherit their variety from the prompt that produced them.
    let diversity =
        diversity::compute(&questions, options.diversity_seed, options.diversity_max_sample);

    let report = SynthesisReport {
        trait_name: trait_name.clone(),
        trait_description: options.trait_description.clone(),
        opposite,
        requested: options.count,
        attempts,
        kept: kept.len(),
        rejected_empty,
        rejected_refusals,
        rejected_duplicates,
        refusal_retries,
        diversity,
    };
    Ok((PairSet { trait_name, pairs: kept }, report))
}

/// Verbatim from the Python generator; the same shape produces both sides,
/// with the trait description swapped for its opposite on the negative.
fn persona_prompt(question: &str, personality: &str) -> String {
    format!(
        "Question: {question}\n\nAnswer the question AS IF you have this personality: {personality}\n\nWrite 1-2 sentences showing this personality clearly. Just the answer."
    )
}

fn resolve_trait_name(options: &SynthesisOptions) -> String {
    let name = options.trait_name.trim();
    if !name.is_empty() {
        return name.to_owned();
    }
    let description = options.trait_description.trim();
    match description.char_indices().nth(TRAIT_NAME_LIMIT) {
        Some((offset, _)) => description[..offset].to_owned(),
        None => description.to_owned(),
    }
}

/// Every call to the model in one synthesis run.
///
/// `Runtime::generate` builds a fresh `LogitsProcessor` from
/// `GenerationOptions::seed` on every call, so a fixed seed would replay the
/// identical continuation for the identical prompt and the run would collapse
/// to a single deduplicated pair. Advancing the seed by the call index makes
/// each call an independent draw while keeping the whole run reproducible
/// from the one seed the caller supplied.
struct Generator<'a> {
    runtime: &'a Runtime,
    options: GenerationOptions,
    calls: u64,
}

impl Generator<'_> {
    fn ask(&mut self, prompt: &str) -> Result<String> {
        let options =
            GenerationOptions { seed: self.options.seed.wrapping_add(self.calls), ..self.options };
        self.calls = self.calls.wrapping_add(1);
        Ok(self.runtime.generate(prompt, None, options)?.trim().to_owned())
    }
}
