//! The tokenizer a fixture declaration names: a tokenizer.json read as it
//! is, or a tokenizer declaration compiled by `ster model tokenizer create`'s
//! compiler, checked against the config's vocabulary size.

use std::{fs, path::Path};

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::failures::{
    fixture::{declaration as causes, read_refusal},
    refuse,
};

use super::spec::{Dimensions, TokenizerInput};

/// The tokenizer bytes `input` names below `parent`, and the compiler's
/// verification for a declared one.
pub(super) fn resolve(
    input: &TokenizerInput,
    parent: &Path,
    dimensions: &Dimensions,
) -> Result<(Vec<u8>, Option<crate::tokenizer::VerificationReport>)> {
    let size = dimensions.vocab_size.get();
    match input {
        TokenizerInput::File(source) => {
            let source = parent.join(source);
            let shown = source.display().to_string();
            let bytes = fs::read(&source).map_err(|error| read_refusal("tokenizer", &source, &error))?;
            let tokenizer = Tokenizer::from_bytes(&bytes).map_err(|error| {
                refuse(
                    &causes::TOKENIZER_UNPARSED,
                    &[("path", &shown), ("error", &error.to_string())],
                )
            })?;
            let vocab = tokenizer.get_vocab(true);
            let Some(highest) = vocab.values().max().copied() else {
                return Err(refuse(&causes::TOKENIZER_EMPTY, &[("path", &shown)]));
            };
            // An id past `usize` is past any vocab_size too.
            let outside = |id: u32| usize::try_from(id).map_or(true, |id| id >= size);
            if let Some((token, id)) = vocab.iter().find(|(_, id)| outside(**id)) {
                return Err(refuse(
                    &causes::TOKEN_OUTSIDE,
                    &[
                        ("path", &shown),
                        ("token", &format!("{token:?}")),
                        ("id", &id.to_string()),
                        ("size", &size.to_string()),
                        ("highest", &highest.to_string()),
                        ("needed", &(u64::from(highest) + 1).to_string()),
                    ],
                ));
            }
            Ok((bytes, None))
        }
        TokenizerInput::Declaration(source) => {
            let source = parent.join(&source.from);
            // The compiler declares its own refusals; they pass through.
            let compiled = crate::tokenizer::compile(&source)?;
            // This producer assigns contiguous model ids by declared order
            // and native added ids after them, so its measured unique count
            // bounds every vocabulary id without parsing the same bytes twice.
            if compiled.vocabulary_size > size {
                return Err(refuse(
                    &causes::TOKENIZER_TOO_WIDE,
                    &[
                        ("path", &source.display().to_string()),
                        ("count", &compiled.vocabulary_size.to_string()),
                        ("size", &size.to_string()),
                    ],
                ));
            }
            Ok((compiled.document.into_bytes(), Some(compiled.verification)))
        }
    }
}
