//! Proving a compiled tokenizer: load the document the library wrote,
//! encode the declaration's verification text, check every encoded token
//! against the vocabulary's own identity for it, and decode the ids back.

use anyhow::Result;
use tokenizers::Tokenizer;
use wisent_errors::Cause;

use super::{VerificationReport, declaration::Verification};
use crate::failures::{refuse, tokenizer as causes};

/// The vocabulary size and the verification report of the tokenizer
/// document `bytes`, `source` naming where it came from (`saved at …`).
pub(super) fn inspect_native(
    bytes: &[u8],
    verification: Verification,
    source: &str,
) -> Result<(usize, VerificationReport)> {
    let text = verification.text.as_str();
    let at = |cause: &'static Cause, extra: &[(&str, &str)]| {
        let mut values = vec![("source", source), ("text", text)];
        values.extend_from_slice(extra);
        refuse(cause, &values)
    };
    let saved = Tokenizer::from_bytes(bytes).map_err(|error| {
        at(&causes::UNLOADED, &[("bytes", &bytes.len().to_string()), ("error", &error.to_string())])
    })?;
    let special = verification.add_special_tokens.to_string();
    let encoding = saved.encode(text, verification.add_special_tokens).map_err(|error| {
        at(&causes::UNENCODED, &[("special", &special), ("error", &error.to_string())])
    })?;
    let tokens = encoding.get_tokens();
    for (position, (id, token)) in encoding.get_ids().iter().zip(tokens).enumerate() {
        let declared = saved.token_to_id(token);
        if declared != Some(*id) {
            let declared = declared.map_or_else(|| "no id".to_owned(), |declared| format!("id {declared}"));
            return Err(at(
                &causes::IDENTITY_MISMATCH,
                &[
                    ("token", token),
                    ("id", &id.to_string()),
                    ("declared", &declared),
                    ("position", &position.to_string()),
                    ("tokens", &tokens.len().to_string()),
                ],
            ));
        }
    }
    let decoded = saved
        .decode(encoding.get_ids(), verification.skip_special_tokens)
        .map_err(|error| {
            at(
                &causes::UNDECODED,
                &[
                    ("ids", &format!("{:?}", encoding.get_ids())),
                    ("skip", &verification.skip_special_tokens.to_string()),
                    ("error", &error.to_string()),
                ],
            )
        })?;
    Ok((
        saved.get_vocab_size(true),
        VerificationReport {
            text: verification.text,
            token_ids: encoding.get_ids().to_vec(),
            tokens: encoding.get_tokens().to_vec(),
            decoded,
        },
    ))
}

