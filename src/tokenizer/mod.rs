//! Compile an ordered native token declaration into a usable WordLevel tokenizer.

mod declaration;

use declaration::{Declaration, Verification};
use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use tokenizers::{Model, Tokenizer, models::wordlevel::WordLevel};

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

fn inspect_native(
    bytes: &[u8],
    verification: Verification,
    context: &str,
) -> Result<(usize, VerificationReport)> {
    let saved = Tokenizer::from_bytes(bytes)
        .map_err(|error| anyhow::anyhow!("load native tokenizer {context}: {error}"))?;
    let encoding = saved
        .encode(verification.text.as_str(), verification.add_special_tokens)
        .map_err(|error| {
            anyhow::anyhow!("encode verification text through tokenizer {context}: {error}")
        })?;
    for (id, token) in encoding.get_ids().iter().zip(encoding.get_tokens()) {
        let declared = saved.token_to_id(token);
        ensure!(
            declared == Some(*id),
            "tokenizer {context} encoded token {token:?} as id {id}, but its vocabulary maps that token to {declared:?}"
        );
    }
    let decoded = saved
        .decode(encoding.get_ids(), verification.skip_special_tokens)
        .map_err(|error| {
            anyhow::anyhow!("decode verification token ids through tokenizer {context}: {error}")
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
        .with_context(|| format!("read tokenizer declaration {}", declaration.display()))?;
    let input: Declaration = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse tokenizer declaration {}", declaration.display()))?;
    ensure!(
        !input.model_tokens.is_empty(),
        "tokenizer declaration {} contains no modelTokens",
        declaration.display()
    );
    ensure!(
        !input.unknown_token.is_empty(),
        "tokenizer declaration requires unknownToken"
    );
    ensure!(
        !input.verification.text.trim().is_empty(),
        "tokenizer declaration requires verification.text"
    );
    ensure!(
        input
            .model_tokens
            .iter()
            .any(|token| token.content == input.unknown_token),
        "tokenizer unknownToken {:?} is absent from its declared modelTokens",
        input.unknown_token
    );
    let vocabulary = input
        .model_tokens
        .iter()
        .enumerate()
        .map(|(index, token)| {
            ensure!(
                !token.content.is_empty(),
                "tokenizer token at position {index} has empty content"
            );
            let id = u32::try_from(index).with_context(|| {
                format!(
                    "tokenizer token at position {index} exceeds the native token id representation"
                )
            })?;
            Ok((token.content.clone(), id))
        })
        .collect::<Result<_>>()?;
    let model = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token(input.unknown_token)
        .build()
        .map_err(|error| anyhow::anyhow!("build native WordLevel tokenizer: {error}"))?;
    if model.get_vocab_size() != input.model_tokens.len() {
        let mut seen = HashSet::new();
        for (index, token) in input.model_tokens.iter().enumerate() {
            if !seen.insert(token.content.as_str()) {
                bail!(
                    "tokenizer token {:?} at position {index} duplicates earlier content; token identities must be unique",
                    token.content
                );
            }
        }
        bail!(
            "native WordLevel vocabulary holds {} identities for {} declared modelTokens",
            model.get_vocab_size(),
            input.model_tokens.len()
        );
    }
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_normalizer(input.pipeline.normalizer);
    tokenizer.with_pre_tokenizer(input.pipeline.pre_tokenizer);
    tokenizer.with_post_processor(input.pipeline.post_processor);
    tokenizer.with_decoder(input.pipeline.decoder);
    for (index, token) in input.added_tokens.iter().enumerate() {
        ensure!(
            !token.content.is_empty(),
            "tokenizer declaration {} has empty addedTokens content at position {index}",
            declaration.display()
        );
    }
    let applied = tokenizer.add_tokens(&input.added_tokens);
    let registered = tokenizer.get_added_vocabulary().get_added_tokens_decoder();
    ensure!(
        applied == input.added_tokens.len() && registered.len() == input.added_tokens.len(),
        "tokenizer declaration {} requested {} added-token records, but the native operation applied {applied} and stored {} unique identities; addedTokens content must be unique",
        declaration.display(),
        input.added_tokens.len(),
        registered.len()
    );
    let document = tokenizer.to_string(true).map_err(|error| {
        anyhow::anyhow!(
            "serialize native tokenizer from {}: {error}",
            declaration.display()
        )
    })?;
    drop(tokenizer);
    drop(input.model_tokens);
    drop(input.added_tokens);
    Ok((document, input.verification))
}

/// Compile and persist a new native tokenizer, then reload and exercise it.
/// Existing output is refused. Native model token IDs are positions in the
/// caller's declaration, never a vocabulary embedded in this implementation.
pub fn create(declaration: &Path, output: &Path) -> Result<TokenizerReport> {
    ensure!(
        output.to_str().is_some(),
        "tokenizer output {} is not representable in the JSON result path",
        output.display()
    );
    let (document, verification) = prepare(declaration)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create tokenizer output parent {}", parent.display()))?;
    }
    // Serialization and input checks precede ownership of the output. The
    // native exclusive open refuses an existing file, directory or symlink.
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .with_context(|| {
            format!(
                "create new tokenizer output {}; existing output is not replaced",
                output.display()
            )
        })?;
    let result = (|| -> Result<TokenizerReport> {
        file.write_all(document.as_bytes())
            .with_context(|| format!("write native tokenizer {}", output.display()))?;
        file.sync_all()
            .with_context(|| format!("persist native tokenizer {}", output.display()))?;
        drop(file);
        let observed = fs::read(output)
            .with_context(|| format!("read back native tokenizer {}", output.display()))?;
        ensure!(
            observed == document.as_bytes(),
            "native tokenizer {} readback differs from the serialized document",
            output.display()
        );
        let (vocabulary_size, verification) = inspect_native(
            &observed,
            verification,
            &format!("saved at {}", output.display()),
        )?;
        Ok(TokenizerReport {
            path: output.to_owned(),
            vocabulary_size,
            verification,
        })
    })();
    match result {
        Ok(report) => Ok(report),
        Err(error) => match fs::remove_file(output) {
            Ok(()) => Err(error.context(format!(
                "tokenizer creation failed; this invocation's output {} was removed",
                output.display()
            ))),
            Err(cleanup) => Err(error.context(format!(
                "tokenizer creation failed and removing its output {} also failed: {cleanup}",
                output.display()
            ))),
        },
    }
}
