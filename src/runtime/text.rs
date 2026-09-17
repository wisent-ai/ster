//! Turning text into the exact token sequence this checkpoint expects,
//! including the conversation format it was post-trained in.

use anyhow::{bail, Result};

use crate::{chat, workflow};

use super::Runtime;

impl Runtime {
    /// Chooses whether this run encodes text through the model's own chat
    /// template, and reports what that resolved to.
    ///
    /// A separate step rather than a loader argument, because the answer
    /// depends on a file only the loader has read by then, and because every
    /// caller that never asks — steering, extraction, pair synthesis, the
    /// frozen judge inside policy optimization — must keep encoding exactly
    /// the bytes it encoded before. The progress line is written here so that
    /// one sentence reaches the operator whichever surface asked, and exactly
    /// once per run.
    pub fn set_chat_template(&mut self, choice: chat::Choice) -> chat::Status {
        self.chat_status = match choice {
            chat::Choice::Off => chat::Status::Off,
            chat::Choice::Auto if self.chat.is_some() => chat::Status::Applied,
            chat::Choice::Auto => chat::Status::Absent,
        };
        workflow::progress(self.chat_status.sentence().to_owned());
        self.chat_status
    }

    /// What this run decided about the chat template.
    pub fn chat_status(&self) -> chat::Status {
        self.chat_status
    }

    /// The template, but only while this run is actually applying it. Every
    /// encoder asks through here, so `off` is one check in one place rather
    /// than a condition each of them could forget.
    pub(super) fn applied_template(&self) -> Option<&chat::Template> {
        match self.chat_status {
            chat::Status::Applied => self.chat.as_ref(),
            chat::Status::Absent | chat::Status::Off => None,
        }
    }

    /// Tokenizes `prompt` and `completion` separately and returns the joined
    /// ids plus the index where the completion begins.
    ///
    /// The split is not cosmetic: it is the only thing that tells the loss
    /// which tokens are the model's answer. Special tokens are added for the
    /// prompt exactly as `encode` adds them, and deliberately not for the
    /// completion — a second begin-of-sequence marker in the middle of the
    /// sequence would be a token the model is asked to predict and never sees
    /// at inference.
    ///
    /// Under a chat template both halves come from the template instead, and
    /// neither is tokenized with the tokenizer's special tokens: the rendered
    /// string already spells every marker the model expects, so asking the
    /// tokenizer to add its own would prepend a second begin-of-sequence. The
    /// boundary still lands exactly where the assistant's own tokens start,
    /// because the completion half is what the template added *after* the
    /// generation prompt — markers included, which is right, since the turn's
    /// end marker is a token the model must learn to emit.
    pub fn encode_example(&self, prompt: &str, completion: &str) -> Result<(Vec<u32>, usize)> {
        if completion.trim().is_empty() {
            bail!("training example has an empty completion");
        }
        let (mut ids, tail) = match self.applied_template() {
            Some(template) => {
                let (head, tail) = template.example(prompt, completion)?;
                (self.tokenize(&head, false, "prompt")?, self.tokenize(&tail, false, "completion")?)
            }
            None => (self.encode(prompt)?, self.tokenize(completion, false, "completion")?),
        };
        if tail.is_empty() {
            bail!("training example completion produced no tokens");
        }
        let boundary = ids.len();
        ids.extend_from_slice(&tail);
        if ids.len() < 2 {
            bail!("training example encodes to fewer than two tokens, so there is nothing to predict");
        }
        Ok((ids, boundary))
    }

    /// One whole utterance, tokenized as the model would have produced it.
    ///
    /// This is what a preference pair's two sides are: complete responses with
    /// no prompt in front of them, which is why they cannot go through
    /// `encode_example`. Under a chat template each side is rendered as an
    /// assistant turn, so the preference is measured over the same markers
    /// inference will put around it; with no template it is `encode`,
    /// unchanged.
    pub fn encode_response(&self, text: &str) -> Result<Vec<u32>> {
        match self.applied_template() {
            Some(template) => {
                let ids = self.tokenize(&template.response(text)?, false, "prompt")?;
                if ids.is_empty() {
                    bail!("tokenizer produced no tokens");
                }
                Ok(ids)
            }
            None => self.encode(text),
        }
    }

    /// Tokenizes one text with the tokenizer's own special tokens, exactly as
    /// every prompt in Ster is tokenized.
    ///
    /// Preference optimization scores whole texts rather than prompt and
    /// completion halves, so it needs this rather than `encode_example`; the
    /// begin-of-sequence marker the tokenizer prepends is what gives the first
    /// real token a position to be predicted from.
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        if prompt.trim().is_empty() {
            bail!("prompt must not be empty");
        }
        let ids = self.tokenize(prompt, true, "prompt")?;
        if ids.is_empty() {
            bail!("tokenizer produced no tokens");
        }
        Ok(ids)
    }

    /// Every id this tokenizer spells `label` with as exactly one token.
    ///
    /// A decision reads the model's next-token distribution at the position
    /// where an answer letter begins, and a vocabulary carries that letter
    /// more than once: bare `A`, SentencePiece's word-initial `▁A`, and
    /// byte-level BPE's `ĠA`. Which one the model puts its mass on depends on
    /// whether the prompt ended in whitespace or a chat marker, so every
    /// spelling the vocabulary holds is scored and the reader sums them. The
    /// vocabulary is consulted directly rather than through `encode`, because
    /// a Llama tokenizer prepends a space to whatever it encodes and would
    /// never hand back the bare form. A label the vocabulary spells with no
    /// single token cannot be read from one position, which is a refusal
    /// rather than a guess.
    pub fn label_tokens(&self, label: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(3);
        for spelling in [label.to_owned(), format!("\u{2581}{label}"), format!("\u{0120}{label}")] {
            if let Some(id) = self.tokenizer.token_to_id(&spelling)
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        if ids.is_empty() {
            bail!("this tokenizer has no single token for the answer label {label:?}, so it cannot be read from one position");
        }
        Ok(ids)
    }

    /// The one call into the tokenizer, so `special` is a decision made at
    /// each site rather than a default. `what` is the noun the refusal names,
    /// which is why it is passed rather than derived: the same call tokenizes
    /// a prompt and a completion and the operator needs to know which failed.
    pub(super) fn tokenize(&self, text: &str, special: bool, what: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, special)
            .map_err(|error| anyhow::anyhow!("failed to tokenize {what}: {error}"))?
            .get_ids()
            .to_vec())
    }
}
