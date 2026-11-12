//! The pieces a built vocabulary is finished with: the stretches its split
//! isolates and the special tokens every sequence starts with.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;
use tokenizers::{
    SplitDelimiterBehavior, Tokenizer,
    pre_tokenizers::split::{Split, SplitPattern},
    processors::template::{Template, TemplateProcessing},
};

use crate::failures::{
    refuse,
    model::tokenizer::{MODEL_UNBUILT, PREFIX_ABSENT, SPLIT_UNCOMPILABLE},
};

/// The refusal of the tokenizers library to build Ster's `model` for the
/// vocabulary at `path` of `rows` tokens, `built` what Ster gave it.
fn unbuilt(model: &str, path: &Path, rows: usize, built: &str, error: &dyn std::fmt::Display) -> anyhow::Error {
    refuse(
        &MODEL_UNBUILT,
        &[
            ("model", model),
            ("path", &path.display().to_string()),
            ("rows", &rows.to_string()),
            ("built", built),
            ("error", &error.to_string()),
        ],
    )
}

/// A post-processor that starts every sequence with the special tokens
/// `prefix`, built from pieces rather than the `"<bos> $A"` text form, which
/// reads a `:` inside a token (PLaMo's `<|plamo:bos|>`) as a type-id
/// separator.
pub(super) fn prefixed(
    tokenizer: &Tokenizer,
    prefix: &[&str],
    path: &Path,
) -> Result<TemplateProcessing> {
    let size = tokenizer.get_vocab_size(true);
    let mut specials = Vec::new();
    for token in prefix {
        let id = tokenizer.token_to_id(token).ok_or_else(|| {
            refuse(
                &PREFIX_ABSENT,
                &[
                    ("path", &path.display().to_string()),
                    ("token", &format!("{token:?}")),
                    ("size", &size.to_string()),
                ],
            )
        })?;
        specials.push((token.to_string(), id));
    }
    let built = format!("prefix {prefix:?}");
    let refused = |error: &dyn std::fmt::Display| unbuilt("prefix template", path, size, &built, error);
    // The pieces are Ster's own: `{"SpecialToken": ..}` for each prefix
    // token, then `{"Sequence": {"id": "A"|"B"}}`, the only ids `Sequence`
    // reads; every special token is in `specials`.
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
        serde_json::from_value(Value::Array(pieces)).map_err(|error| refused(&error))
    };
    TemplateProcessing::builder()
        .single(template(&["A"])?)
        .pair(template(&["A", "B"])?)
        .special_tokens(specials)
        .build()
        .map_err(|error| refused(&error))
}

/// A pre-tokenizer that cuts every match of `pattern` out as its own
/// stretch, or the tokenizers library's refusal to compile it.
fn split(pattern: &str) -> tokenizers::Result<tokenizers::PreTokenizerWrapper> {
    let regex = SplitPattern::Regex(pattern.to_owned());
    Ok(Split::new(regex, SplitDelimiterBehavior::Isolated, false)?.into())
}

/// The stretches PLaMo surrounds with boundary characters: every match of
/// `pattern`, built from the whole-number threshold `key` states. The
/// regular expression engine the tokenizers library splits with bounds a
/// repeat count, so a threshold past its bound is refused.
pub(super) fn isolate(
    pattern: &str,
    key: &str,
    threshold: u64,
    path: &Path,
) -> Result<tokenizers::PreTokenizerWrapper> {
    split(pattern).map_err(|error| {
        refuse(
            &SPLIT_UNCOMPILABLE,
            &[
                ("path", &path.display().to_string()),
                ("key", key),
                ("threshold", &threshold.to_string()),
                ("pattern", &format!("{pattern:?}")),
                ("error", &error.to_string()),
            ],
        )
    })
}

/// A tiktoken class's own split, `pattern` one of the constants
/// `tiktoken::GLM4_PATTERN` and `tiktoken::KIMI_PATTERN`, for the
/// vocabulary at `path` of `rows` ranks.
pub(super) fn class_split(
    pattern: &'static str,
    path: &Path,
    rows: usize,
) -> Result<tokenizers::PreTokenizerWrapper> {
    split(pattern).map_err(|error| unbuilt("split pattern", path, rows, &format!("pattern {pattern:?}"), &error))
}
