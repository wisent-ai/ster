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

mod pieces;
mod tiktoken;

use std::{fs, path::Path};

use anyhow::Result;
use pieces::{isolate, prefixed};
use serde_json::Value;
use tiktoken::{GLM4_PATTERN, GLM4_PREFIX, KIMI_PATTERN, KIMI_RESERVED_SPECIAL_TOKENS, tiktoken};
use tokenizers::{
    AddedToken, Tokenizer,
    decoders::{byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence as DecoderSequence},
    models::unigram::Unigram,
    pre_tokenizers::sequence::Sequence as PreTokenizerSequence,
};

use crate::failures::{
    file, refuse,
    model::tokenizer::{
        BOS_UNNAMED, CLASS_UNNAMED, CLASS_UNSUPPORTED, CONFIG_MALFORMED, JSON_MALFORMED,
        JSON_UNLOADABLE, MODEL_UNBUILT, ROW_NOT_JSON, ROW_UNSCORED, VOCABULARY_EMPTY,
        json_refusal, read_refusal,
    },
};

/// The vocabulary file at `path` as text, refused as empty when it holds
/// no row of `form`.
fn vocabulary_text(path: &Path, form: &str) -> Result<String> {
    let text = fs::read_to_string(path).map_err(|error| read_refusal("vocabulary", path, &error))?;
    if text.lines().all(|line| line.trim().is_empty()) {
        return Err(refuse(
            &VOCABULARY_EMPTY,
            &[
                ("path", &path.display().to_string()),
                ("state", &file::state(path)),
                ("form", form),
            ],
        ));
    }
    Ok(text)
}

/// The refusal `cause` of line `line` (`text`) of vocabulary `path`, which
/// is not a row of `form`, with `values` the cause's sentence names beside
/// them.
fn row_refusal(
    cause: &'static wisent_errors::Cause,
    (path, line, form): (&Path, usize, &str),
    text: &str,
    values: &[(&str, &str)],
) -> anyhow::Error {
    let path = path.display().to_string();
    let (line, text) = (line.to_string(), format!("{text:?}"));
    let mut all = vec![
        ("path", path.as_str()),
        ("line", line.as_str()),
        ("form", form),
        ("text", text.as_str()),
    ];
    all.extend_from_slice(values);
    refuse(cause, &all)
}

/// Entry `index` of a vocabulary row as JSON, or `absent`.
fn entry(row: &Value, index: usize) -> String {
    row.get(index).map_or("absent".to_owned(), Value::to_string)
}

/// The tokenizer config at `config` as JSON.
pub(crate) fn tokenizer_config(config: &Path) -> Result<Value> {
    let bytes = fs::read(config).map_err(|error| read_refusal("tokenizer config", config, &error))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| json_refusal(&CONFIG_MALFORMED, "tokenizer config", config, &error))
}

/// Transformers' `tokenizer.json` at `path`: refused as not JSON, or as
/// JSON the tokenizers library does not read as a tokenizer.
fn transformers(path: &Path) -> Result<Tokenizer> {
    let text = fs::read_to_string(path).map_err(|error| read_refusal("tokenizer", path, &error))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|error| json_refusal(&JSON_MALFORMED, "tokenizer", path, &error))?;
    text.parse::<Tokenizer>().map_err(|error| {
        let keys = match value.as_object() {
            Some(object) => object.keys().cloned().collect::<Vec<_>>().join(", "),
            None => format!("none, it is {value}"),
        };
        refuse(
            &JSON_UNLOADABLE,
            &[
                ("path", &path.display().to_string()),
                ("keys", &keys),
                ("error", &error.to_string()),
            ],
        )
    })
}

/// The tokenizer at `path`: `tokenizer.jsonl` read as PLaMo's, a
/// `tokenizer.model` as GLM-4's and a `tiktoken.model` as Kimi's tiktoken
/// vocabulary, anything else as Transformers' `tokenizer.json`.
/// `tokenizer_config` supplies PLaMo's split thresholds, whether a sequence
/// starts with its BOS, and the tiktoken vocabularies' special tokens.
pub fn load(path: &Path, tokenizer_config_path: Option<&Path>) -> Result<Tokenizer> {
    // Any other name, `tokenizer.json` among them, is Transformers' form.
    let shown = path.display().to_string();
    let Some(name @ ("tokenizer.jsonl" | "tokenizer.model" | "tiktoken.model")) =
        path.file_name().and_then(|name| name.to_str())
    else {
        return transformers(path);
    };
    let settings: Value = match tokenizer_config_path {
        Some(config) => tokenizer_config(config)?,
        None => Value::Null,
    };
    if name == "tokenizer.jsonl" {
        return plamo(path, &settings);
    }
    // A tiktoken vocabulary is read only by the classes whose split pattern
    // Ster knows; any other (SentencePiece's protobuf, another pattern) is
    // refused by the class its config names.
    let config = match tokenizer_config_path {
        Some(config) => config.display().to_string(),
        None => "absent".to_owned(),
    };
    let stated = |pointer: &str| match settings.pointer(pointer) {
        Some(value) => value.to_string(),
        None => "absent".to_owned(),
    };
    let Some(class) = settings
        .pointer("/auto_map/AutoTokenizer/0")
        .or_else(|| settings.get("tokenizer_class"))
        .and_then(Value::as_str)
    else {
        return Err(refuse(
            &CLASS_UNNAMED,
            &[
                ("path", &shown),
                ("config", &config),
                ("auto_map", &stated("/auto_map/AutoTokenizer")),
                ("class", &stated("/tokenizer_class")),
            ],
        ));
    };
    let class_name = match class.rsplit_once('.') {
        Some((_, name)) => name,
        None => class,
    };
    match class_name {
        // GLM-4 reserves no special ids beyond those its config names
        // (`ChatGLM4Tokenizer` in
        // https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/tokenization_chatglm.py).
        "ChatGLM4Tokenizer" => {
            tiktoken(path, (&settings, &config), GLM4_PATTERN, 0, &GLM4_PREFIX)
        }
        "TikTokenTokenizer" => tiktoken(
            path,
            (&settings, &config),
            KIMI_PATTERN,
            KIMI_RESERVED_SPECIAL_TOKENS,
            &[],
        ),
        _ => Err(refuse(
            &CLASS_UNSUPPORTED,
            &[
                ("path", &shown),
                ("class", &format!("{class:?}")),
                ("config", &config),
            ],
        )),
    }
}

fn plamo(path: &Path, settings: &Value) -> Result<Tokenizer> {
    const FORM: &str = "[piece, score, kind]";
    let text = vocabulary_text(path, FORM)?;
    let mut vocabulary = Vec::new();
    let mut unknown = None;
    let mut controls = Vec::new();
    let mut bytes = Vec::new();
    for (id, line) in text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let row: Value = serde_json::from_str(line).map_err(|error| {
            let error = error.to_string();
            row_refusal(&ROW_NOT_JSON, (path, id + 1, FORM), line, &[("error", &error)])
        })?;
        let (Some(piece), Some(score)) = (
            row.get(0).and_then(Value::as_str),
            row.get(1).and_then(Value::as_f64),
        ) else {
            return Err(row_refusal(
                &ROW_UNSCORED,
                (path, id + 1, FORM),
                line,
                &[("first", &entry(&row, 0)), ("second", &entry(&row, 1))],
            ));
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
    let rows = vocabulary.len();
    let model = Unigram::from(vocabulary, unknown, true).map_err(|error| {
        let unknown = unknown.map_or("none".to_owned(), |id| format!("row {}", id + 1));
        refuse(
            &MODEL_UNBUILT,
            &[
                ("model", "unigram model"),
                ("path", &path.display().to_string()),
                ("rows", &rows.to_string()),
                ("built", &format!("unknown piece: {unknown}")),
                ("error", &error.to_string()),
            ],
        )
    })?;
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
        splits.push(isolate(&pattern, "break_around_repeated_chars_threshold", repeats, path)?);
    }
    if let Some(spaces) = threshold("break_around_consecutive_spaces_threshold") {
        let pattern = format!(" {{{spaces},}}");
        splits.push(isolate(&pattern, "break_around_consecutive_spaces_threshold", spaces, path)?);
    }
    if !splits.is_empty() {
        tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(splits)));
    }
    if settings.get("add_bos_token").and_then(Value::as_bool) == Some(true) {
        let Some(bos) = settings.get("bos_token").and_then(Value::as_str) else {
            let named = settings.get("bos_token").map_or("absent".to_owned(), Value::to_string);
            return Err(refuse(
                &BOS_UNNAMED,
                &[("path", &path.display().to_string()), ("bos", &named)],
            ));
        };
        let processor = prefixed(&tokenizer, &[bos], path)?;
        tokenizer.with_post_processor(Some(processor));
    }
    Ok(tokenizer)
}
