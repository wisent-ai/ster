//! A checkpoint's tokenizer: Transformers' `tokenizer.json`, or PLaMo's
//! `tokenizer.jsonl` (`Plamo3Tokenizer` in PLaMo's `tokenization_plamo.py`),
//! one `[piece, score, kind]` row per token id. PLaMo's tokenizer is a
//! unigram model: the split of each stretch of text whose pieces' scores sum
//! highest, a character no piece covers spelled as its UTF-8 bytes
//! (`<0xXX>`), and its control tokens and the stretches its
//! `break_around_*` thresholds isolate never crossed by a piece. It is built
//! here as the same `tokenizers` Unigram model with byte fallback.

use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use tokenizers::{
    AddedToken, Tokenizer,
    decoders::{byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence as DecoderSequence},
    models::unigram::Unigram,
    pre_tokenizers::{
        sequence::Sequence as PreTokenizerSequence,
        split::{Split, SplitPattern},
    },
    processors::template::{Template, TemplateProcessing},
    SplitDelimiterBehavior,
};

/// The tokenizer at `path`: `tokenizer.jsonl` read as PLaMo's, anything
/// else as Transformers' `tokenizer.json`. `tokenizer_config` supplies
/// PLaMo's split thresholds and whether a sequence starts with its BOS.
pub fn load(path: &Path, tokenizer_config: Option<&Path>) -> Result<Tokenizer> {
    if path.extension().and_then(|extension| extension.to_str()) != Some("jsonl") {
        return Tokenizer::from_file(path).map_err(|error| anyhow!("failed to load tokenizer {}: {error}", path.display()));
    }
    let settings: Value = match tokenizer_config {
        Some(config) => serde_json::from_slice(&fs::read(config).with_context(|| format!("failed to read {}", config.display()))?)
            .with_context(|| format!("invalid tokenizer config {}", config.display()))?,
        None => Value::Null,
    };
    plamo(path, &settings)
}

fn plamo(path: &Path, settings: &Value) -> Result<Tokenizer> {
    let text = fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut vocabulary = Vec::new();
    let mut unknown = None;
    let mut controls = Vec::new();
    let mut bytes = Vec::new();
    for (id, line) in text.lines().filter(|line| !line.trim().is_empty()).enumerate() {
        let row: Value = serde_json::from_str(line).with_context(|| format!("{} line {} is not JSON", path.display(), id + 1))?;
        let (Some(piece), Some(score)) = (row.get(0).and_then(Value::as_str), row.get(1).and_then(Value::as_f64)) else {
            bail!("{} line {} is not a [piece, score, kind] row", path.display(), id + 1);
        };
        match row.get(2).and_then(Value::as_str) {
            Some("UNKNOWN") => {
                unknown = Some(id);
                controls.push(piece.to_owned());
            }
            Some("CONTROL") => controls.push(piece.to_owned()),
            Some("BYTE") => bytes.push(id),
            _ => {}
        }
        vocabulary.push((piece.to_owned(), score));
    }
    // A byte token is only ever a fallback in PLaMo's split, never a piece
    // matching its own spelling; it is given the lowest score any piece has,
    // so a spelled-out `<0xXX>` in the text is split as text.
    let lowest = vocabulary.iter().map(|(_, score)| *score).fold(f64::INFINITY, f64::min);
    for id in bytes {
        vocabulary[id].1 = lowest;
    }
    let model = Unigram::from(vocabulary, unknown, true).map_err(|error| anyhow!("{} is not a unigram vocabulary: {error}", path.display()))?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(DecoderSequence::new(vec![ByteFallback::new().into(), Fuse::new().into()])));
    let specials: Vec<AddedToken> = controls.iter().map(|piece| AddedToken::from(piece.clone(), true)).collect();
    tokenizer.add_special_tokens(&specials);
    let threshold = |key: &str| settings.get(key).and_then(Value::as_u64).filter(|threshold| *threshold > 0);
    let mut splits = Vec::new();
    if let Some(repeats) = threshold("break_around_repeated_chars_threshold") {
        let pattern = format!("(.)\\1{{{},}}", repeats - 1);
        splits.push(isolate(&pattern)?);
    }
    if let Some(spaces) = threshold("break_around_consecutive_spaces_threshold") {
        splits.push(isolate(&format!(" {{{spaces},}}"))?);
    }
    if !splits.is_empty() {
        tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(splits)));
    }
    if settings.get("add_bos_token").and_then(Value::as_bool) == Some(true) {
        let bos = settings.get("bos_token").and_then(Value::as_str).context("the tokenizer config adds a BOS token it does not name")?;
        let id = tokenizer.token_to_id(bos).with_context(|| format!("the BOS token {bos:?} is not in {}", path.display()))?;
        // Built from pieces rather than the `"<bos> $A"` text form, which
        // reads the `:` in PLaMo's `<|plamo:bos|>` as a type-id separator.
        let template = |sequences: &[&str]| -> Result<Template> {
            let mut pieces = vec![serde_json::json!({ "SpecialToken": { "id": bos, "type_id": 0 } })];
            pieces.extend(sequences.iter().map(|id| serde_json::json!({ "Sequence": { "id": id, "type_id": 0 } })));
            serde_json::from_value(Value::Array(pieces)).context("invalid BOS template")
        };
        let processor = TemplateProcessing::builder()
            .single(template(&["A"])?)
            .pair(template(&["A", "B"])?)
            .special_tokens(vec![(bos.to_owned(), id)])
            .build()
            .map_err(|error| anyhow!("invalid BOS template: {error}"))?;
        tokenizer.with_post_processor(Some(processor));
    }
    Ok(tokenizer)
}

/// A pre-tokenizer that cuts every match of `pattern` out as its own
/// stretch, as PLaMo surrounds it with boundary characters.
fn isolate(pattern: &str) -> Result<tokenizers::PreTokenizerWrapper> {
    let regex = SplitPattern::Regex(pattern.to_owned());
    Ok(Split::new(regex, SplitDelimiterBehavior::Isolated, false)
        .map_err(|error| anyhow!("invalid split pattern {pattern:?}: {error}"))?
        .into())
}
