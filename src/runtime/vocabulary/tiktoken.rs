//! Tiktoken vocabularies: GLM-4's `tokenizer.model` and Kimi's
//! `tiktoken.model`, built as a byte-level BPE.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;
use tokenizers::{
    AddedToken, Tokenizer, pre_tokenizers::sequence::Sequence as PreTokenizerSequence,
};

use crate::failures::{
    model::tokenizer::{
        ADDED_CONTENT_ABSENT, ADDED_KEY_NOT_ID, ADDED_NOT_OBJECT, MODEL_UNBUILT, RANK_NOT_ID,
        ROW_SPACELESS, SPECIAL_OUTSIDE, TOKEN_NOT_BASE64,
    },
    refuse,
};

use super::{
    pieces::{class_split, prefixed},
    row_refusal, vocabulary_text,
};

/// GLM-4's split of text into pieces before byte-pair merging
/// (`ChatGLM4Tokenizer.pat_str` in
/// https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/tokenization_chatglm.py).
pub(super) const GLM4_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The special tokens `ChatGLM4Tokenizer.get_prefix_tokens` puts before
/// every sequence it encodes with special tokens
/// (https://huggingface.co/THUDM/glm-4-9b-chat/blob/main/tokenization_chatglm.py).
pub(super) const GLM4_PREFIX: [&str; 2] = ["[gMASK]", "<sop>"];

/// Kimi's split of text into pieces (`TikTokenTokenizer.pat_str` in
/// https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/tokenization_kimi.py);
/// `&&` is a character class intersection, which the Oniguruma engine
/// `tokenizers` splits with reads as tiktoken's does.
pub(super) const KIMI_PATTERN: &str = r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// How many special-token ids follow Kimi's ranks
/// (`TikTokenTokenizer.num_reserved_special_tokens` in
/// https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/tokenization_kimi.py);
/// those `added_tokens_decoder` does not name are `<|reserved_token_{id}|>`.
pub(super) const KIMI_RESERVED_SPECIAL_TOKENS: usize = 256;

/// A tiktoken vocabulary (GLM-4's `tokenizer.model`, `ChatGLM4Tokenizer`;
/// Kimi's `tiktoken.model`, `TikTokenTokenizer`): one `base64(bytes) rank`
/// line per token id, read as the byte-level BPE tiktoken runs — each piece
/// of `pattern` kept whole when it is a token, else merged pair by pair in
/// rank order, the merges recovered from the ranks as Transformers'
/// `TikTokenConverter` recovers them — with the special tokens after it:
/// `added_tokens_decoder`'s, at least `reserved` of them, unnamed ones
/// spelled `<|reserved_token_{id}|>`; and `prefix` before every sequence.
/// `config` is where `settings` were read, as a refusal names it.
pub(super) fn tiktoken(
    path: &Path,
    (settings, config): (&Value, &str),
    pattern: &'static str,
    reserved: usize,
    prefix: &[&str],
) -> Result<Tokenizer> {
    use base64::Engine;
    const FORM: &str = "`base64 rank`";
    let text = vocabulary_text(path, FORM)?;
    let mut ranks: std::collections::HashMap<Vec<u8>, u32> = std::collections::HashMap::new();
    for (line_number, line) in text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let line_number = line_number + 1;
        let row = (path, line_number, FORM);
        let Some((token, rank)) = line.split_once(' ') else {
            let characters = line.chars().count().to_string();
            return Err(row_refusal(&ROW_SPACELESS, row, line, &[("characters", &characters)]));
        };
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(token)
            .map_err(|error| {
                let (token, error) = (format!("{token:?}"), error.to_string());
                row_refusal(&TOKEN_NOT_BASE64, row, line, &[("token", &token), ("error", &error)])
            })?;
        let rank = rank.trim().parse::<u32>().map_err(|error| {
            let (rank, error) = (format!("{rank:?}"), error.to_string());
            let largest = u32::MAX.to_string();
            row_refusal(
                &RANK_NOT_ID,
                row,
                line,
                &[("rank", &rank), ("largest", &largest), ("error", &error)],
            )
        })?;
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
    let merge_count = merges.len();
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
        // tokenizers 0.22's `BpeBuilder::build` errs only on a merge whose
        // halves or join its vocabulary lacks, and each merge above joins
        // two ranked halves into a ranked token; a library that reads them
        // otherwise is refused by name.
        .map_err(|error| {
            refuse(
                &MODEL_UNBUILT,
                &[
                    ("model", "byte-level BPE"),
                    ("path", &path.display().to_string()),
                    ("rows", &ranks.len().to_string()),
                    ("built", &format!("merges recovered from the ranks: {merge_count}")),
                    ("error", &error.to_string()),
                ],
            )
        })?;
    let mut tokenizer = Tokenizer::new(model);
    let byte_level = tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false);
    tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(vec![
        class_split(pattern, path, ranks.len())?,
        byte_level.clone().into(),
    ])));
    tokenizer.with_decoder(Some(byte_level));
    let given = added_tokens(path, (settings, config))?;
    let base = ranks.len() as u64;
    let count = (reserved as u64).max(given.len() as u64);
    if let Some((id, _)) = given
        .iter()
        .find(|(id, _)| !(base..base + count).contains(*id))
    {
        return Err(refuse(
            &SPECIAL_OUTSIDE,
            &[
                ("path", &path.display().to_string()),
                ("id", &id.to_string()),
                ("count", &count.to_string()),
                ("base", &base.to_string()),
                ("end", &(base + count).to_string()),
            ],
        ));
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

/// The special tokens the tokenizer config of the vocabulary at `path`
/// names in `added_tokens_decoder`, by id. A config without the key adds
/// none: Transformers starts its added tokens empty and fills them only
/// from an `added_tokens_decoder` the config holds
/// (`PreTrainedTokenizerBase._from_pretrained` in
/// https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py).
fn added_tokens(
    path: &Path,
    (settings, config): (&Value, &str),
) -> Result<std::collections::BTreeMap<u64, String>> {
    let refused = |cause: &'static wisent_errors::Cause, key: &str, value: &Value| {
        refuse(
            cause,
            &[
                ("path", &path.display().to_string()),
                ("config", config),
                ("key", &format!("{key:?}")),
                ("value", &value.to_string()),
            ],
        )
    };
    let mut given = std::collections::BTreeMap::new();
    let added = match settings.get("added_tokens_decoder") {
        None => return Ok(given),
        Some(Value::Object(added)) => added,
        Some(other) => return Err(refused(&ADDED_NOT_OBJECT, "added_tokens_decoder", other)),
    };
    for (id, token) in added {
        let Ok(number) = id.parse::<u64>() else {
            return Err(refused(&ADDED_KEY_NOT_ID, id, token));
        };
        let Some(content) = token.get("content").and_then(Value::as_str) else {
            return Err(refused(&ADDED_CONTENT_ABSENT, id, token));
        };
        given.insert(number, content.to_owned());
    }
    Ok(given)
}

/// GPT-2's printable spelling of each byte (`bytes_to_unicode` in
/// https://github.com/openai/gpt-2/blob/master/src/encoder.py), which
/// byte-level BPE vocabularies are written in: the printable bytes stand for
/// themselves and the rest are moved, in order, to code points from 256.
fn byte_spelling() -> Vec<char> {
    // https://github.com/openai/gpt-2/blob/master/src/encoder.py
    let printable = |byte: u8| matches!(byte, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut moved = 0u32;
    (0..=u8::MAX)
        .map(|byte| {
            if printable(byte) {
                char::from(byte)
            } else {
                moved += 1;
                // `moved` counts the bytes `printable` leaves out, at most
                // 256, so `u8::MAX + moved` is below 512, never a surrogate.
                char::from_u32(u32::from(u8::MAX) + moved)
                    .expect("a code point below 512 is a char")
            }
        })
        .collect()
}
