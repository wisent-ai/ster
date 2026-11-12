//! Compile an ordered native token declaration into a usable WordLevel tokenizer.

mod declaration;
mod verify;

use declaration::{Declaration, Verification};
use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use serde::Serialize;
use tokenizers::{Model, Tokenizer, models::wordlevel::WordLevel};
use wisent_errors::Cause;

use crate::failures::{amend, file, refuse, tokenizer as causes};
use verify::inspect_native;

/// What the declaration is, for the `ster.file` refusals that name it.
const ROLE: &str = "the tokenizer declaration";

#[derive(Debug, Serialize)]
pub struct VerificationReport {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub decoded: String,
}

#[derive(Debug, Serialize)]
pub struct TokenizerReport {
    pub path: PathBuf,
    pub vocabulary_size: usize,
    pub verification: VerificationReport,
}

pub struct Compilation {
    pub document: String,
    pub vocabulary_size: usize,
    pub verification: VerificationReport,
}

/// Compile and exercise a declaration without writing an intermediate file.
/// Fixture creation uses this same producer for a declared tokenizer source.
pub fn compile(declaration: &Path) -> Result<Compilation> {
    let (document, verification) = prepare(declaration)?;
    let (vocabulary_size, verification) = inspect_native(
        document.as_bytes(),
        verification,
        &format!("compiled from {}", declaration.display()),
    )?;
    Ok(Compilation {
        document,
        vocabulary_size,
        verification,
    })
}

fn prepare(declaration: &Path) -> Result<(String, Verification)> {
    let bytes = fs::read(declaration)
        .map_err(|error| file::read::refusal_or(&causes::MISSING, ROLE, declaration, &error))?;
    let input: Declaration = serde_json::from_slice(&bytes)
        .map_err(|error| file::json::refusal(ROLE, causes::FORM, declaration, &error))?;
    let path = declaration.display().to_string();
    let count = input.model_tokens.len().to_string();
    let added = input.added_tokens.len().to_string();
    let at = |cause: &'static Cause, extra: &[(&str, &str)]| {
        let mut values = vec![("path", path.as_str()), ("count", count.as_str()), ("added", added.as_str())];
        values.extend_from_slice(extra);
        refuse(cause, &values)
    };
    if input.model_tokens.is_empty() {
        return Err(at(&causes::MODEL_TOKENS_NONE, &[]));
    }
    if input.unknown_token.is_empty() {
        return Err(at(&causes::UNKNOWN_BLANK, &[]));
    }
    if input.verification.text.trim().is_empty() {
        let bytes = input.verification.text.len().to_string();
        return Err(at(&causes::VERIFICATION_BLANK, &[("bytes", &bytes)]));
    }
    let unknown = input.unknown_token.as_str();
    if !input.model_tokens.iter().any(|token| token.content == unknown) {
        let first = format!("\"{}\"", input.model_tokens[0].content);
        return Err(at(&causes::UNKNOWN_UNDECLARED, &[("unknown", unknown), ("first", &first)]));
    }
    // Positions are the model's ids: every content once, none empty, every
    // position a 32-bit id.
    let mut first_at: HashMap<&str, usize> = HashMap::with_capacity(input.model_tokens.len());
    for (position, token) in input.model_tokens.iter().enumerate() {
        let shown = position.to_string();
        if token.content.is_empty() {
            let before = match position.checked_sub(1) {
                Some(earlier) => format!("\"{}\"", input.model_tokens[earlier].content),
                None => "none, it is the first".to_owned(),
            };
            return Err(at(&causes::MODEL_TOKEN_BLANK, &[("position", &shown), ("before", &before)]));
        }
        if u32::try_from(position).is_err() {
            let limit = (u64::from(u32::MAX) + 1).to_string();
            return Err(at(&causes::IDS_EXHAUSTED, &[("position", &shown), ("limit", &limit)]));
        }
        if let Some(first) = first_at.insert(token.content.as_str(), position) {
            return Err(at(
                &causes::MODEL_TOKEN_REPEATED,
                &[("content", &token.content), ("position", &shown), ("first", &first.to_string())],
            ));
        }
    }
    drop(first_at);
    // Every position fits in 32 bits and every content is unique, so the
    // vocabulary holds one identity per declared token.
    let vocabulary = input
        .model_tokens
        .iter()
        .zip(0u32..)
        .map(|(token, id)| (token.content.clone(), id))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token(input.unknown_token.clone())
        .build()
        .expect("WordLevelBuilder::build reads a file only when one is named, and none is");
    debug_assert_eq!(model.get_vocab_size(), input.model_tokens.len());
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_normalizer(input.pipeline.normalizer);
    tokenizer.with_pre_tokenizer(input.pipeline.pre_tokenizer);
    tokenizer.with_post_processor(input.pipeline.post_processor);
    tokenizer.with_decoder(input.pipeline.decoder);
    if let Some(position) = input.added_tokens.iter().position(|token| token.content.is_empty()) {
        return Err(at(&causes::ADDED_TOKEN_BLANK, &[("position", &position.to_string())]));
    }
    let applied = tokenizer.add_tokens(&input.added_tokens);
    let stored = tokenizer.get_added_vocabulary().get_added_tokens_decoder().len();
    if applied != input.added_tokens.len() || stored != input.added_tokens.len() {
        let mut seen: HashMap<&str, usize> = HashMap::new();
        let repeated: Vec<String> = input
            .added_tokens
            .iter()
            .enumerate()
            .filter_map(|(position, token)| {
                seen.insert(token.content.as_str(), position)
                    .map(|first| format!("\"{}\" at {first} and {position}", token.content))
            })
            .collect();
        let repeated = if repeated.is_empty() {
            "none by content; the library merged records of one identity".to_owned()
        } else {
            repeated.join(", ")
        };
        return Err(at(
            &causes::ADDED_TOKENS_MERGED,
            &[
                ("requested", &added),
                ("applied", &applied.to_string()),
                ("stored", &stored.to_string()),
                ("repeated", &repeated),
            ],
        ));
    }
    let document = tokenizer.to_string(true).map_err(|error| {
        let pipeline = format!(
            "normalizer {}, preTokenizer {}, postProcessor {}, decoder {}",
            stage(tokenizer.get_normalizer().is_some()),
            stage(tokenizer.get_pre_tokenizer().is_some()),
            stage(tokenizer.get_post_processor().is_some()),
            stage(tokenizer.get_decoder().is_some()),
        );
        at(&causes::UNSERIALIZED, &[("pipeline", &pipeline), ("error", &error.to_string())])
    })?;
    Ok((document, input.verification))
}

/// How a pipeline stage was declared, for a refusal that lists them.
fn stage(declared: bool) -> &'static str {
    if declared { "declared" } else { "null" }
}

/// Compile and persist a new native tokenizer, then reload and exercise it.
/// Existing output is refused. Native model token IDs are positions in the
/// caller's declaration, never a vocabulary embedded in this implementation.
pub fn create(declaration: &Path, output: &Path) -> Result<TokenizerReport> {
    let shown = output.display().to_string();
    if output.to_str().is_none() {
        return Err(refuse(&causes::OUTPUT_NOT_UTF8, &[("path", &shown)]));
    }
    let (document, verification) = prepare(declaration)?;
    let role = "the tokenizer ster model tokenizer create writes";
    if let Some(parent) = output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|error| {
            file::write::refusal("creating the directory of", role, output, "nothing was written", &error)
        })?;
    }
    // Serialization and input checks precede ownership of the output. The
    // native exclusive open refuses an existing file, directory or symlink.
    let mut created = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|error| {
            if error.kind() == ErrorKind::AlreadyExists {
                refuse(&causes::OUTPUT_EXISTS, &[("path", &shown), ("state", &file::state(output))])
            } else {
                file::write::refusal("creating", role, output, "nothing was written", &error)
            }
        })?;
    let left = "the file this run created is removed next";
    let result = (|| -> Result<TokenizerReport> {
        created
            .write_all(document.as_bytes())
            .map_err(|error| file::write::refusal("writing", role, output, left, &error))?;
        created
            .sync_all()
            .map_err(|error| file::write::refusal("syncing to disk", role, output, left, &error))?;
        drop(created);
        let observed = fs::read(output)
            .map_err(|error| file::read::refusal("the tokenizer just written", output, &error))?;
        if observed != document.as_bytes() {
            let offset = observed
                .iter()
                .zip(document.as_bytes())
                .position(|(read, written)| read != written)
                .unwrap_or_else(|| observed.len().min(document.len()));
            return Err(refuse(
                &causes::READBACK_DIFFERS,
                &[
                    ("path", &shown),
                    ("written", &document.len().to_string()),
                    ("read", &observed.len().to_string()),
                    ("offset", &offset.to_string()),
                ],
            ));
        }
        let (vocabulary_size, verification) =
            inspect_native(&observed, verification, &format!("saved at {shown}"))?;
        Ok(TokenizerReport {
            path: output.to_owned(),
            vocabulary_size,
            verification,
        })
    })();
    result.map_err(|error| match fs::remove_file(output) {
        Ok(()) => amend(error, format!("the output {shown} this run created was removed")),
        Err(cleanup) => amend(
            error,
            format!(
                "removing the output {shown} this run created also failed ({}: {cleanup}); it is {}, and `rm {shown}` removes it",
                file::kind(&cleanup),
                file::state(output)
            ),
        ),
    })
}
