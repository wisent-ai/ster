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
//! BPE (see [`tiktoken`]).

use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
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
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
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
    let class = settings
        .pointer("/auto_map/AutoTokenizer/0")
        .or_else(|| settings.get("tokenizer_class"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let class_name = class.rsplit('.').next().unwrap_or_default();
    match class_name {
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

/// GLM-4's split of text into pieces before byte-pair merging
/// (`tokenization_chatglm.py`, `ChatGLM4Tokenizer.pat_str`).
const GLM4_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The special tokens `ChatGLM4Tokenizer.get_prefix_tokens` puts before
/// every sequence it encodes with special tokens.
const GLM4_PREFIX: [&str; 2] = ["[gMASK]", "<sop>"];

/// Kimi's split of text into pieces (`tokenization_kimi.py`,
/// `TikTokenTokenizer.pat_str`); `&&` is a character class intersection,
/// which the Oniguruma engine `tokenizers` splits with reads as tiktoken's
/// does.
const KIMI_PATTERN: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// How many special-token ids follow Kimi's ranks
/// (`TikTokenTokenizer.num_reserved_special_tokens` in
/// `tokenization_kimi.py`); those `added_tokens_decoder` does not name are
/// `<|reserved_token_{id}|>`.
const KIMI_RESERVED_SPECIAL_TOKENS: usize = 256;

/// A tiktoken vocabulary (GLM-4's `tokenizer.model`, `ChatGLM4Tokenizer`;
/// Kimi's `tiktoken.model`, `TikTokenTokenizer`): one `base64(bytes) rank`
/// line per token id, read as the byte-level BPE tiktoken runs — each piece
/// of `pattern` kept whole when it is a token, else merged pair by pair in
/// rank order, the merges recovered from the ranks as Transformers'
/// `TikTokenConverter` recovers them — with the special tokens after it:
/// `added_tokens_decoder`'s, at least `reserved` of them, unnamed ones
/// spelled `<|reserved_token_{id}|>`; and `prefix` before every sequence.
fn tiktoken(
    path: &Path,
    settings: &Value,
    pattern: &str,
    reserved: usize,
    prefix: &[&str],
) -> Result<Tokenizer> {
    use base64::Engine;
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut ranks: std::collections::HashMap<Vec<u8>, u32> = std::collections::HashMap::new();
    for (line_number, line) in text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let parsed = line.split_once(' ').and_then(|(token, rank)| {
            Some((
                base64::engine::general_purpose::STANDARD
                    .decode(token)
                    .ok()?,
                rank.trim().parse::<u32>().ok()?,
            ))
        });
        let Some((bytes, rank)) = parsed else {
            bail!(
                "{} line {} is not a `base64 rank` pair",
                path.display(),
                line_number + 1
            );
        };
        ranks.insert(bytes, rank);
    }
    let spell = byte_spelling();
    let spelled =
        |bytes: &[u8]| -> String { bytes.iter().map(|byte| spell[*byte as usize]).collect() };
    let mut merges: Vec<(u32, String, String)> = Vec::new();
    for (token, rank) in &ranks {
        let mut local: Vec<(u32, u32, String, String)> = (1..token.len())
            .filter_map(|cut| {
                let (left, right) = token.split_at(cut);
                Some((
                    *ranks.get(left)?,
                    *ranks.get(right)?,
                    spelled(left),
                    spelled(right),
                ))
            })
            .collect();
        local.sort();
        merges.extend(
            local
                .into_iter()
                .map(|(_, _, left, right)| (*rank, left, right)),
        );
    }
    merges.sort_by_key(|(rank, _, _)| *rank);
    let vocabulary: tokenizers::models::bpe::Vocab = ranks
        .iter()
        .map(|(bytes, rank)| (spelled(bytes), *rank))
        .collect();
    let model = tokenizers::models::bpe::BPE::builder()
        .vocab_and_merges(
            vocabulary,
            merges
                .into_iter()
                .map(|(_, left, right)| (left, right))
                .collect(),
        )
        .ignore_merges(true)
        .build()
        .map_err(|error| anyhow!("{} is not a byte-pair vocabulary: {error}", path.display()))?;
    let mut tokenizer = Tokenizer::new(model);
    let byte_level = tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(vec![
        isolate(pattern)?,
        byte_level.clone().into(),
    ])));
    tokenizer.with_decoder(Some(byte_level));
    let given: std::collections::BTreeMap<u64, String> = settings
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .map(|added| {
            added
                .iter()
                .filter_map(|(id, token)| {
                    Some((id.parse().ok()?, token.get("content")?.as_str()?.to_owned()))
                })
                .collect()
        })
        .unwrap_or_default();
    let base = ranks.len() as u64;
    let count = (reserved as u64).max(given.len() as u64);
    if let Some((id, _)) = given
        .iter()
        .find(|(id, _)| !(base..base + count).contains(*id))
    {
        bail!(
            "{} numbers a special token {id}, outside the {count} ids after its {base} ranks; Ster adds them in order",
            path.display()
        );
    }
    let mut added = Vec::new();
    for id in base..base + count {
        let content = match given.get(&id) {
            Some(content) => content.clone(),
            None => format!("<|reserved_token_{id}|>"),
        };
        added.push(AddedToken::from(content, true));
    }
    tokenizer.add_special_tokens(&added);
    if !prefix.is_empty() {
        let processor = prefixed(&tokenizer, prefix, path)?;
        tokenizer.with_post_processor(Some(processor));
    }
    Ok(tokenizer)
}

/// GPT-2's printable spelling of each byte (`bytes_to_unicode` in its
/// `encoder.py`), which byte-level BPE vocabularies are written in: the
/// printable bytes stand for themselves and the rest are moved, in order,
/// to code points from 256.
fn byte_spelling() -> Vec<char> {
    let printable = |byte: u8| matches!(byte, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut moved = 0u32;
    (0..=u8::MAX)
        .map(|byte| {
            if printable(byte) {
                char::from(byte)
            } else {
                moved += 1;
                char::from_u32(u32::from(u8::MAX) + moved)
                    .expect("a code point below 512 is a char")
            }
        })
        .collect()
}
