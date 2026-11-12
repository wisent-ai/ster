//! Writing a contrastive pair set with a model, driven by whichever generator
//! the caller picked: one opposite-trait description, then a question and its
//! two answers per attempt, each answer judged for refusal as it lands.
//! Steering needs hidden states and stays local; writing pair text does not,
//! so the writer may be a hosted route.

use anyhow::Result;

use super::generator::Generator;
use crate::{
    artifact::{ContrastivePair, PairSet},
    failures::{pairs::synthesize as declared, refuse},
    pairs::quality::dedupe,
    pairs::quality::{diversity, refusal},
    workflow,
};

mod options;

pub use options::{SynthesisOptions, SynthesisReport};

/// Verbatim from the Python generator: the single instruction that produces
/// the user question each pair answers.
const QUESTION_INSTRUCTION: &str = "Write one short question a user might ask. Example: 'What is your favorite hobby?' Just the question, nothing else.";

/// Verbatim `roleplay_neg_fix` from Wisent's `db_instructions/mini_dp.py`.
/// The Python cleaner sends it as a system message; `Runtime::generate` takes
/// a single prompt string, so it is prepended to the user turn instead.
const ROLEPLAY_NEG_FIX: &str = "You are fixing ONLY the negative example of a contrastive pair.\nProduce a single concise negative response for the given prompt that exemplifies the UNDESIRED trait.\nIt must be fictional/hypothetical, safe, and non-actionable. Return raw text only.";

/// How one attempt at a pair ended.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The question or an answer came back empty.
    Empty,
    /// The positive refused its trait, or the negative still refused after its repair.
    Refused,
    /// The pair repeated one already kept.
    Duplicate,
    Kept,
}

/// One attempt and whether its negative had to be asked again; every count in
/// the report is read off the list of these, so nothing is counted by hand.
struct Attempt {
    outcome: Outcome,
    repaired_negative: bool,
}

/// Generate a contrastive pair set with `generator`.
///
/// Asks once for the opposite trait unless the caller stated it, then per
/// attempt asks for a question and for a positive and a negative answer to
/// it. A refusing positive drops the attempt; a refusing negative is asked
/// once more with `ROLEPLAY_NEG_FIX` and dropped if it still refuses; a pair
/// that repeats a kept one is dropped. Each pair is cleaned as it lands, so
/// the loop stops the moment `count` surviving pairs exist, or when
/// `count × retry_multiplier` attempts are spent. Wisent's Python generator
/// instead re-cleans the whole set every ten pairs; the prompt texts above
/// are its, verbatim, because they are the contract the pairs must match.
pub fn synthesize(
    generator: Generator<'_>,
    options: &SynthesisOptions,
) -> Result<(PairSet, SynthesisReport)> {
    options.validate()?;

    let trait_name = resolve_trait_name(options);
    // Every prompt asked so far, in order: how many there are is the call
    // index each new call's seed advances by.
    let mut asked: Vec<String> = Vec::new();
    let ask = |prompt: &str, asked: &mut Vec<String>| -> Result<String> {
        let text = generator.generate(prompt, options, u64::try_from(asked.len())?)?;
        asked.push(prompt.to_owned());
        Ok(text)
    };

    // The opposite is the caller's, or the generator's answer to one
    // question; a generator that names none leaves the negatives nothing to
    // be, so the run is refused rather than given an opposite nobody stated.
    let opposite = match options
        .opposite
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(value) => value.to_owned(),
        None => {
            let answer = ask(
                &format!(
                    "What is the OPPOSITE personality trait of: {}?\n\nDescribe the opposite in one sentence, be specific about what words/style/tone to use.",
                    options.trait_description
                ),
                &mut asked,
            )?;
            if answer.is_empty() {
                return Err(refuse(
                    &declared::OPPOSITE_EMPTY,
                    &[
                        ("generator", &generator.label()),
                        ("trait", &format!("{:?}", options.trait_description)),
                        ("tokens", &options.generation.max_new_tokens.to_string()),
                        ("temperature", &options.generation.temperature.to_string()),
                    ],
                ));
            }
            answer
        }
    };

    // Named once, before the loop: a streamed run has to state which model is
    // writing and which opposite trait the negative side is answering as,
    // and neither changes for the rest of the run.
    workflow::progress(format!(
        "synthesizing with {} against opposite trait {:?}",
        generator.label(),
        opposite
    ));

    let mut index = dedupe::Index::new(options.dedupe)?;
    let mut kept: Vec<ContrastivePair> = Vec::with_capacity(options.count.get());
    let mut questions: Vec<String> = Vec::with_capacity(options.count.get());
    let max_attempts = options
        .count
        .get()
        .saturating_mul(options.retry_multiplier.get());
    let mut attempts: Vec<Attempt> = Vec::new();

    while kept.len() < options.count.get() && attempts.len() < max_attempts {
        workflow::progress(format!(
            "synthesizing: {} of {} pairs kept after {} attempts",
            kept.len(),
            options.count,
            attempts.len()
        ));
        let mut attempt = Attempt {
            outcome: Outcome::Kept,
            repaired_negative: false,
        };
        attempt.outcome = 'attempt: {
            let question = ask(QUESTION_INSTRUCTION, &mut asked)?;
            if question.is_empty() {
                break 'attempt Outcome::Empty;
            }
            let positive = ask(
                &persona_prompt(&question, &options.trait_description),
                &mut asked,
            )?;
            if positive.is_empty() {
                break 'attempt Outcome::Empty;
            }
            let mut negative = ask(&persona_prompt(&question, &opposite), &mut asked)?;
            if negative.is_empty() {
                break 'attempt Outcome::Empty;
            }

            // Whether an answer declined its role is the generator's judgement,
            // asked once per answer and read in one exact form.
            let refuses =
                |personality: &str, answer: &str, asked: &mut Vec<String>| -> Result<bool> {
                    let text = ask(
                        &refusal::judgement_prompt(&question, personality, answer),
                        asked,
                    )?;
                    refusal::verdict(&generator.label(), &text)
                };

            // A refusing positive has no repair path: the trait itself is what the
            // model declined to roleplay, so re-asking would refuse again.
            if refuses(&options.trait_description, &positive, &mut asked)? {
                break 'attempt Outcome::Refused;
            }

            // `RefusalerCleaner` + `BaseRefusaler::fix_negative`: one re-prompt,
            // never a loop, and a still-refusing replacement drops the pair.
            if refuses(&opposite, &negative, &mut asked)? {
                attempt.repaired_negative = true;
                let replacement = ask(
                    &format!(
                        "{ROLEPLAY_NEG_FIX}\n\nPrompt: {question}\nTrait label: {trait_name}\nTrait description: {opposite}"
                    ),
                    &mut asked,
                )?;
                if replacement.is_empty() || refuses(&opposite, &replacement, &mut asked)? {
                    break 'attempt Outcome::Refused;
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
                break 'attempt Outcome::Duplicate;
            }
            questions.push(question);
            kept.push(pair);
            Outcome::Kept
        };
        attempts.push(attempt);
    }

    // Diversity is measured over the questions, matching the Python report:
    // the answers inherit their variety from the prompt that produced them.
    let diversity = diversity::compute(&questions);
    let counted = |outcome: Outcome| {
        attempts
            .iter()
            .filter(|attempt| attempt.outcome == outcome)
            .count()
    };

    let report = SynthesisReport {
        generator: generator.label(),
        trait_name: trait_name.clone(),
        trait_description: options.trait_description.clone(),
        opposite,
        requested: options.count.get(),
        attempts: attempts.len(),
        kept: kept.len(),
        rejected_empty: counted(Outcome::Empty),
        rejected_refusals: counted(Outcome::Refused),
        rejected_duplicates: counted(Outcome::Duplicate),
        refusal_retries: attempts
            .iter()
            .filter(|attempt| attempt.repaired_negative)
            .count(),
        diversity,
    };
    // A run that kept nothing has no set to write; the counts say why, where
    // writing would only say that the set holds no pairs.
    if kept.is_empty() {
        return Err(refuse(
            &declared::NO_PAIRS,
            &[
                ("attempts", &report.attempts.to_string()),
                ("generator", &report.generator),
                ("empty", &report.rejected_empty.to_string()),
                ("refusals", &report.rejected_refusals.to_string()),
                ("duplicates", &report.rejected_duplicates.to_string()),
            ],
        ));
    }
    Ok((
        PairSet {
            trait_name,
            pairs: kept,
        },
        report,
    ))
}

/// Verbatim from the Python generator; the same shape produces both sides,
/// with the trait description swapped for its opposite on the negative.
fn persona_prompt(question: &str, personality: &str) -> String {
    format!(
        "Question: {question}\n\nAnswer the question AS IF you have this personality: {personality}\n\nWrite 1-2 sentences showing this personality clearly. Just the answer."
    )
}

/// The set's trait name: the one given, else the trait's description itself.
fn resolve_trait_name(options: &SynthesisOptions) -> String {
    match options.trait_name.as_deref().map(str::trim) {
        Some(name) if !name.is_empty() => name.to_owned(),
        _ => options.trait_description.trim().to_owned(),
    }
}
