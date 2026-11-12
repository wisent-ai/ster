//! Turning text into the exact token sequence this checkpoint expects,
//! including the conversation format it was post-trained in.

use anyhow::Result;

use crate::failures::{
    refuse,
    runtime::encode::{
        CONVERSATION_UNTEMPLATED, LABEL_UNSPELLED, PIECE_UNKNOWN, SYSTEM_UNPLACEABLE, TEXT_BLANK,
        TOKENIZER_FAILED, TOKENS_NONE,
    },
};
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
    /// once per run. A template that does not compile is refused here, under
    /// `auto` only: `off` runs the checkpoint on raw text whatever its
    /// template says.
    pub fn set_chat_template(&mut self, choice: chat::Choice) -> Result<chat::Status> {
        if matches!(choice, chat::Choice::Auto) && self.chat.is_err() {
            // The refusal is raised once, by the run that asked for the
            // template; the runtime keeps no template in its place.
            if let Err(refusal) = std::mem::replace(&mut self.chat, Ok(None)) {
                return Err(refusal);
            }
        }
        self.chat_status = match (choice, &self.chat) {
            (chat::Choice::Off, _) => chat::Status::Off,
            (chat::Choice::Auto, Ok(Some(_))) => chat::Status::Applied,
            (chat::Choice::Auto, Ok(None) | Err(_)) => chat::Status::Absent,
        };
        workflow::progress(self.chat_status.sentence().to_owned());
        Ok(self.chat_status)
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
            // `set_chat_template` marks the template applied only after
            // reading it as `Ok(Some(_))`.
            chat::Status::Applied => match &self.chat {
                Ok(template) => template.as_ref(),
                Err(refusal) => unreachable!(
                    "the chat template is marked applied yet it was refused: {refusal}"
                ),
            },
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
    ///
    /// `system` is the example's system turn. Raw text has no place for one,
    /// so an example carrying it on a run without an applied template is a
    /// refusal naming what the run decided, not a guess at the markers.
    pub fn encode_example(
        &self,
        system: Option<&str>,
        prompt: &str,
        completion: &str,
    ) -> Result<(Vec<u32>, usize)> {
        let (mut ids, tail) = match (self.applied_template(), system) {
            (Some(template), _) => {
                let (head, tail) = template.example(system, prompt, completion)?;
                (
                    self.tokens(&head, false, "prompt")?,
                    self.tokens(&tail, false, "completion")?,
                )
            }
            (None, Some(system)) => {
                let status = self.chat_status.label();
                return Err(refuse(
                    &SYSTEM_UNPLACEABLE,
                    &[("system", &format!("{system:?}")), ("status", status)],
                ));
            }
            (None, None) => (
                self.encode(prompt)?,
                self.tokens(completion, false, "completion")?,
            ),
        };
        // Both halves hold a token, so the first completion token always has
        // one before it to be predicted from.
        let boundary = ids.len();
        ids.extend_from_slice(&tail);
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
            Some(template) => self.tokens(&template.response(text)?, false, "response"),
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
            return Err(refuse(
                &TEXT_BLANK,
                &[("what", "prompt"), ("text", &format!("{prompt:?}"))],
            ));
        }
        self.tokens(prompt, true, "prompt")
    }

    /// A whole conversation, rendered by the model's chat template and ending
    /// where the assistant's next turn begins.
    ///
    /// Only a template knows how turns are delimited, so a conversation has
    /// no raw-text form: with the template absent or turned off this is a
    /// refusal naming what the run decided, not a guess at the markers.
    pub fn encode_conversation(&self, messages: &[chat::Message<'_>]) -> Result<Vec<u32>> {
        let Some(template) = self.applied_template() else {
            return Err(refuse(
                &CONVERSATION_UNTEMPLATED,
                &[("status", self.chat_status.label())],
            ));
        };
        self.tokens(&template.render(messages, true)?, false, "conversation")
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
        let spellings = [
            label.to_owned(),
            format!("\u{2581}{label}"),
            format!("\u{0120}{label}"),
        ];
        for spelling in &spellings {
            if let Some(id) = self.tokenizer.token_to_id(spelling)
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        if ids.is_empty() {
            let looked: Vec<String> = spellings
                .iter()
                .map(|spelling| format!("{spelling:?}"))
                .collect();
            return Err(refuse(
                &LABEL_UNSPELLED,
                &[
                    ("label", &format!("{label:?}")),
                    ("spellings", &looked.join(", ")),
                ],
            ));
        }
        Ok(ids)
    }

    /// The one call into the tokenizer, so `special` is a decision made at
    /// each site rather than a default. `what` is the noun the refusal names,
    /// which is why it is passed rather than derived: the same call tokenizes
    /// a prompt and a completion and the operator needs to know which failed.
    pub(super) fn tokenize(&self, text: &str, special: bool, what: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, special).map_err(|error| {
            let said = error.to_string();
            // The tokenizers library has no error type to tell its reasons
            // apart; a model that meets a piece outside its vocabulary with
            // no unknown token to stand in says so in these words.
            let cause = if said.contains("unk_id") {
                &PIECE_UNKNOWN
            } else {
                &TOKENIZER_FAILED
            };
            refuse(
                cause,
                &[
                    ("what", what),
                    ("text", &format!("{text:?}")),
                    (
                        "vocabulary",
                        &self.tokenizer.get_vocab_size(true).to_string(),
                    ),
                    ("error", &said),
                ],
            )
        })?;
        Ok(encoding.get_ids().to_vec())
    }

    /// [`Self::tokenize`], refusing a text the tokenizer turns into no
    /// tokens: every caller of this one hands the ids to the model, which
    /// reads nothing from an empty sequence.
    fn tokens(&self, text: &str, special: bool, what: &str) -> Result<Vec<u32>> {
        let ids = self.tokenize(text, special, what)?;
        if ids.is_empty() {
            return Err(refuse(
                &TOKENS_NONE,
                &[("what", what), ("text", &format!("{text:?}"))],
            ));
        }
        Ok(ids)
    }
}
