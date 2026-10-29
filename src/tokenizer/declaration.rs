use std::fmt;

use serde::{
    Deserialize, Deserializer,
    de::{SeqAccess, Visitor},
};
use tokenizers::{
    AddedToken, decoders::DecoderWrapper, normalizers::NormalizerWrapper,
    pre_tokenizers::PreTokenizerWrapper, processors::PostProcessorWrapper,
};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Declaration {
    pub(super) unknown_token: String,
    pub(super) model_tokens: Vec<ModelToken>,
    #[serde(deserialize_with = "added_tokens")]
    pub(super) added_tokens: Vec<AddedToken>,
    pub(super) pipeline: Pipeline,
    pub(super) verification: Verification,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelToken {
    pub(super) content: String,
}

// External AddedToken schema identities, retained rather than counted as
// removed classifiers. No caller-supplied ID or unknown flag may disappear.
// https://docs.rs/tokenizers/0.22.2/tokenizers/struct.AddedToken.html
#[derive(Deserialize)]
#[serde(remote = "AddedToken", deny_unknown_fields)]
struct NativeAddedToken {
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

fn added_tokens<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<AddedToken>, D::Error> {
    #[derive(Deserialize)]
    struct Token(#[serde(with = "NativeAddedToken")] AddedToken);
    struct Tokens;
    impl<'de> Visitor<'de> for Tokens {
        type Value = Vec<AddedToken>;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an array of complete native added-token declarations")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
            let mut tokens = Vec::new();
            while let Some(Token(token)) = sequence.next_element()? {
                tokens.push(token);
            }
            Ok(tokens)
        }
    }
    deserializer.deserialize_seq(Tokens)
}

// Each native stage must be stated, including explicit null when not wanted.
// An ordinary Option field would also accept omission.
fn required_stage<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Pipeline {
    #[serde(deserialize_with = "required_stage")]
    pub(super) normalizer: Option<NormalizerWrapper>,
    #[serde(deserialize_with = "required_stage")]
    pub(super) pre_tokenizer: Option<PreTokenizerWrapper>,
    #[serde(deserialize_with = "required_stage")]
    pub(super) post_processor: Option<PostProcessorWrapper>,
    #[serde(deserialize_with = "required_stage")]
    pub(super) decoder: Option<DecoderWrapper>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Verification {
    pub(super) text: String,
    pub(super) add_special_tokens: bool,
    pub(super) skip_special_tokens: bool,
}
