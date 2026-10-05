//! One rollout: the policy's turns in a conversation that starts from a
//! prompt, with a second model writing the user's turns in between.
//!
//! A single-turn run is the degenerate rollout — one assistant turn and no
//! user model — and goes through the same function, so the two cannot drift.
//!
//! The simulated user is the second model answering the same conversation
//! with the roles swapped: its own "assistant" turn is the user's next
//! message. That is the whole trick, and it needs nothing from the user model
//! but a chat template. A reply that is empty once trimmed is the simulated
//! user ending the conversation; the rollout stops there, and the shorter
//! conversation has fewer assistant turns to earn reward from.

use anyhow::{Result, bail};

use crate::{
    chat::{self, Message},
    runtime::{Completion, DeviceChoice, GenerationOptions, Precision, Runtime},
    workflow,
};

/// One played conversation: the policy's turns with the context each was
/// sampled from, and the whole transcript as `(role, text)` in order.
pub struct Rollout {
    pub turns: Vec<Completion>,
    pub transcript: Vec<(&'static str, String)>,
}

/// The model that writes the user's turns. Frozen and generation-only: it is
/// never scored with a gradient and carries no adapter.
pub struct UserSimulator {
    runtime: Runtime,
}

impl UserSimulator {
    /// Loads the user model and requires its chat template, because the
    /// role swap is meaningless without the markers that delimit turns.
    pub fn load(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        precision: Precision,
    ) -> Result<Self> {
        workflow::progress(format!("loading the simulated user {model}"));
        let mut runtime = Runtime::load_at(model, revision, device, precision)?;
        if runtime.set_chat_template(chat::Choice::Auto)? != chat::Status::Applied {
            bail!(
                "the simulated user {model} publishes no chat template, so it cannot write a user's turn in a conversation"
            );
        }
        Ok(Self { runtime })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Speaker {
    User,
    Assistant,
}

impl Speaker {
    fn role(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }

    fn swapped(self) -> Self {
        match self {
            Self::User => Self::Assistant,
            Self::Assistant => Self::User,
        }
    }
}

/// The policy's turns, in order. Each completion's `prompt` is the whole
/// conversation before that turn, so scoring it scores the turn in context.
/// `ster converse` plays the same function with no gradient anywhere, so a
/// conversation evaluated is a conversation trained on.
pub fn rollout(
    runtime: &Runtime,
    user: Option<&UserSimulator>,
    prompt: &str,
    turns: usize,
    generation: GenerationOptions,
    limit: usize,
    draw: &mut u64,
) -> Result<Rollout> {
    let Some(user) = user else {
        let completion = runtime.sample(prompt, None, next(generation, draw))?;
        refuse_empty(&completion, 1)?;
        let transcript = vec![
            ("user", prompt.to_owned()),
            ("assistant", completion.text.clone()),
        ];
        return Ok(Rollout {
            turns: vec![completion],
            transcript,
        });
    };

    let mut said: Vec<(Speaker, String)> = vec![(Speaker::User, prompt.to_owned())];
    let mut played = Vec::with_capacity(turns);
    for turn in 1..=turns {
        let context = runtime.encode_conversation(&messages(&said, false))?;
        let longest = context.len() + generation.max_new_tokens;
        if longest > limit {
            bail!(
                "turn {turn} needs {} conversation tokens plus {} sampled tokens, past the {limit} token limit; lower --turns or raise --max-sequence",
                context.len(),
                generation.max_new_tokens
            );
        }
        let completion = runtime.sample_tokens(context, None, next(generation, draw))?;
        refuse_empty(&completion, turn)?;
        said.push((Speaker::Assistant, completion.text.clone()));
        played.push(completion);
        if turn == turns {
            break;
        }

        let context = user.runtime.encode_conversation(&messages(&said, true))?;
        let reply = user
            .runtime
            .sample_tokens(context, None, next(generation, draw))?;
        if reply.text.trim().is_empty() {
            break;
        }
        said.push((Speaker::User, reply.text));
    }
    let transcript = said
        .into_iter()
        .map(|(speaker, text)| (speaker.role(), text))
        .collect();
    Ok(Rollout {
        turns: played,
        transcript,
    })
}

/// The conversation as one side sees it: as written for the policy, or with
/// every role swapped for the simulated user.
fn messages(said: &[(Speaker, String)], swap: bool) -> Vec<Message<'_>> {
    said.iter()
        .map(|(speaker, content)| {
            let speaker = if swap { speaker.swapped() } else { *speaker };
            Message {
                role: speaker.role(),
                content,
            }
        })
        .collect()
}

/// Every draw in the run gets its own seed, advanced from the operator's.
fn next(generation: GenerationOptions, draw: &mut u64) -> GenerationOptions {
    let advanced = GenerationOptions {
        seed: generation.seed.wrapping_add(*draw),
        ..generation
    };
    *draw = draw.wrapping_add(1);
    advanced
}

/// An assistant turn whose first token was the end of sequence has nothing to
/// take a gradient through, and dropping it would bias the group's baseline
/// upward by removing its worst member, so the whole group is refused.
fn refuse_empty(completion: &Completion, turn: usize) -> Result<()> {
    if completion.tokens.is_empty() {
        bail!("assistant turn {turn} of a sampled conversation was empty");
    }
    Ok(())
}
