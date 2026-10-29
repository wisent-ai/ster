//! A checkpoint's tokenizer: Transformers' `tokenizer.json`, PLaMo's
//! `tokenizer.jsonl`, GLM-4's tiktoken `tokenizer.model` or Kimi's tiktoken
//! `tiktoken.model`.
//!
//! PLaMo's (`Plamo3Tokenizer` in `tokenization_plamo.py`) is one
//! `[piece, score, kind]` row per token id, a unigram model: the split of
//! each stretch of text whose pieces' scores sum highest, a character no
//! piece covers spelled as its UTF-8 bytes (`<0xXX>`), and its control
//! tokens and the stretches its `break_around_*` thresholds isolate never
//! crossed by a piece. It is built here as the same `tokenizers` Unigram
//! model with byte fallback. GLM-4's and Kimi's are built as a byte-level
//! BPE (see [`tiktoken()`]).

mod tiktoken;

use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use tiktoken::{GLM4_PATTERN, GLM4_PREFIX, KIMI_PATTERN, KIMI_RESERVED_SPECIAL_TOKENS, tiktoken};
use tokenizers::{
    AddedToken, SplitDelimiterBehavior, Tokenizer,
    decoders::{byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence as DecoderSequence},
    models::unigram::Unigram,
    pre_tokenizers::{
        sequence::Sequence as PreTokenizerSequence,
        split::{Split, SplitPattern},
    },
    processors::template::{Template, TemplateProcessing},
};

/// The tokenizer at `path`: `tokenizer.jsonl` read as PLaMo's, a
/// `tokenizer.model` as GLM-4's and a `tiktoken.model` as Kimi's tiktoken
/// vocabulary, anything else as Transformers' `tokenizer.json`.
/// `tokenizer_config` supplies PLaMo's split thresholds, whether a sequence
/// starts with its BOS, and the tiktoken vocabularies' special tokens.
pub fn load(path: &Path, tokenizer_config: Option<&Path>) -> Result<Tokenizer> {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        bail!(
            "the tokenizer path {} has no UTF-8 file name; point it at the checkpoint's tokenizer.json, tokenizer.jsonl, tokenizer.model or tiktoken.model",
            path.display()
        );
    };
    if !matches!(
        name,
        "tokenizer.jsonl" | "tokenizer.model" | "tiktoken.model"
    ) {
        return Tokenizer::from_file(path)
            .map_err(|error| anyhow!("failed to load tokenizer {}: {error}", path.display()));
    }
    let settings: Value = match tokenizer_config {
        Some(config) => serde_json::from_slice(
            &fs::read(config).with_context(|| format!("failed to read {}", config.display()))?,
        )
        .with_context(|| format!("invalid tokenizer config {}", config.display()))?,
        None => Value::Null,
    };
    if name == "tokenizer.jsonl" {
        return plamo(path, &settings);
    }
    // A tiktoken vocabulary is read only by the classes whose split pattern
    // Ster knows; any other (SentencePiece's protobuf, another pattern) is
    // refused by the class its config names.
    let Some(class) = settings
        .pointer("/auto_map/AutoTokenizer/0")
        .or_else(|| settings.get("tokenizer_class"))
        .and_then(Value::as_str)
    else {
        bail!(
            "{} is a tiktoken or SentencePiece vocabulary whose tokenizer config names no auto_map.AutoTokenizer or tokenizer_class string, so Ster cannot tell GLM-4's (ChatGLM4Tokenizer) from Kimi's (TikTokenTokenizer); keep the checkpoint's tokenizer_config.json beside it or use a checkpoint that publishes tokenizer.json",
            path.display()
        );
    };
    let class_name = match class.rsplit_once('.') {
        Some((_, name)) => name,
        None => class,
    };
    match class_name {
        // GLM-4 reserves no special ids beyond those its config names
        // (`ChatGLM4Tokenizer` in
        // https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/tokenization_chatglm.py).
        "ChatGLM4Tokenizer" => tiktoken(path, &settings, GLM4_PATTERN, 0, &GLM4_PREFIX),
        "TikTokenTokenizer" => tiktoken(
            path,
            &settings,
            KIMI_PATTERN,
            KIMI_RESERVED_SPECIAL_TOKENS,
            &[],
        ),
        _ => bail!(
            "{} is a tiktoken or SentencePiece vocabulary for {class:?}, and Ster reads one only as GLM-4's (ChatGLM4Tokenizer) or Kimi's (TikTokenTokenizer); use a checkpoint that publishes tokenizer.json",
            path.display()
        ),
    }
}

fn plamo(path: &Path, settings: &Value) -> Result<Tokenizer> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut vocabulary = Vec::new();
    let mut unknown = None;
    let mut controls = Vec::new();
    let mut bytes = Vec::new();
    for (id, line) in text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let row: Value = serde_json::from_str(line)
            .with_context(|| format!("{} line {} is not JSON", path.display(), id + 1))?;
        let (Some(piece), Some(score)) = (
            row.get(0).and_then(Value::as_str),
            row.get(1).and_then(Value::as_f64),
        ) else {
            bail!(
                "{} line {} is not a [piece, score, kind] row",
                path.display(),
                id + 1
            );
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
    let lowest = vocabulary
        .iter()
        .map(|(_, score)| *score)
        .fold(f64::INFINITY, f64::min);
    for id in bytes {
        vocabulary[id].1 = lowest;
    }
    let model = Unigram::from(vocabulary, unknown, true)
        .map_err(|error| anyhow!("{} is not a unigram vocabulary: {error}", path.display()))?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(DecoderSequence::new(vec![
        ByteFallback::new().into(),
        Fuse::new().into(),
    ])));
    let specials: Vec<AddedToken> = controls
        .iter()
        .map(|piece| AddedToken::from(piece.clone(), true))
        .collect();
    tokenizer.add_special_tokens(&specials);
    let threshold = |key: &str| {
        settings
            .get(key)
            .and_then(Value::as_u64)
            .filter(|threshold| *threshold > 0)
    };
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
        let bos = settings
            .get("bos_token")
            .and_then(Value::as_str)
            .context("the tokenizer config adds a BOS token it does not name")?;
        let processor = prefixed(&tokenizer, &[bos], path)?;
        tokenizer.with_post_processor(Some(processor));
    }
    Ok(tokenizer)
}

/// A post-processor that starts every sequence with the special tokens
/// `prefix`, built from pieces rather than the `"<bos> $A"` text form, which
/// reads a `:` inside a token (PLaMo's `<|plamo:bos|>`) as a type-id
/// separator.
fn prefixed(tokenizer: &Tokenizer, prefix: &[&str], path: &Path) -> Result<TemplateProcessing> {
    let mut specials = Vec::new();
    for token in prefix {
        let id = tokenizer
            .token_to_id(token)
            .with_context(|| format!("the prefix token {token:?} is not in {}", path.display()))?;
        specials.push((token.to_string(), id));
    }
    let template = |sequences: &[&str]| -> Result<Template> {
        let mut pieces: Vec<Value> = prefix
            .iter()
            .map(|id| serde_json::json!({ "SpecialToken": { "id": id, "type_id": 0 } }))
            .collect();
        pieces.extend(
            sequences
                .iter()
                .map(|id| serde_json::json!({ "Sequence": { "id": id, "type_id": 0 } })),
        );
        serde_json::from_value(Value::Array(pieces)).context("invalid prefix template")
    };
    TemplateProcessing::builder()
        .single(template(&["A"])?)
        .pair(template(&["A", "B"])?)
        .special_tokens(specials)
        .build()
        .map_err(|error| anyhow!("invalid prefix template: {error}"))
}

/// A pre-tokenizer that cuts every match of `pattern` out as its own
/// stretch, as PLaMo surrounds it with boundary characters.
fn isolate(pattern: &str) -> Result<tokenizers::PreTokenizerWrapper> {
    let regex = SplitPattern::Regex(pattern.to_owned());
    Ok(Split::new(regex, SplitDelimiterBehavior::Isolated, false)
        .map_err(|error| anyhow!("invalid split pattern {pattern:?}: {error}"))?
        .into())
}
