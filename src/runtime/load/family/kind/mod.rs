//! Which decoder family a config names: the families Ster implements and how a
//! config's `model_type` (or its `architectures` class) picks one.

use serde_json::Value;

mod all;
mod names;

/// A decoder layout Ster implements, one per Transformers `model_type`.
///
/// The config's `model_type` field names the layout; [`Family::of`] reads it
/// and [`Family::ALL`] is what the refusal of any other value lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::runtime::load) enum Family {
    Llama,
    Mistral,
    Mixtral,
    Qwen2,
    Qwen2Moe,
    Qwen3,
    Qwen3Moe,
    Phi,
    Phi3,
    Granite,
    GraniteMoe,
    StableLm,
    Starcoder2,
    Cohere,
    Cohere2,
    Nemotron,
    Olmo,
    Olmo2,
    Olmo3,
    Olmoe,
    Exaone4,
    InternLm3,
    SeedOss,
    Arcee,
    Ernie45,
    MiniCpm,
    Orion,
    Glm,
    Glm4,
    GptNeox,
    GptJ,
    SmolLm3,
    Gpt2,
    GptBigCode,
    Opt,
    Bloom,
    Falcon,
    Mpt,
    DeepseekV2,
    DeepseekV3,
    MiniCpm3,
    Mamba,
    FalconMamba,
    Glm4Moe,
    InternLm2,
    Exaone,
    Jamba,
    HunYuanDense,
    Mamba2,
    Bamba,
    GptOss,
    Lfm2,
    Ernie45Moe,
    Dbrx,
    Phimoe,
    HunYuanMoe,
    Telechat,
    Lfm2Moe,
    GraniteMoeHybrid,
    NemotronH,
    Jais2,
    BailingMoe,
    TeleFlm,
    FalconH1,
    Qwen3Next,
    KimiLinear,
    MinimaxM2,
    Step3Text,
    MinimaxText,
    Minimax,
    Zamba2,
    Gemma,
    Gemma2,
    Gemma3Text,
    Gemma4Text,
    Gemma4UnifiedText,
    Mimo,
    Mellum,
    FlexOlmo,
    GraniteSwa,
    GraniteMoeSwa,
    GraniteMoeShared,
    Glm4MoeLite,
    IQuestCoder,
    HyperClovaX,
    VaultGemma,
    Apertus,
    ExaoneMoe,
    Solar,
    Step1,
    TeleChat3,
    Param2Moe,
    PanguEmbedded,
    OlmoHybrid,
    HyV3,
    Nanbeige,
    IQuestLoopCoder,
    NemotronNas,
    DeepseekV32,
    GlmMoeDsa,
    Axk1,
    BailingHybrid,
    MimoV2Flash,
    MimoV2,
    Step3p5,
    K2Horizon,
    PanguUltraMoe,
    LongcatFlash,
    SarvamMoe,
    SarvamMla,
    Cohere2Moe,
    DeepseekMoe,
    Ministral3,
    Llama4Text,
    Qwen35Text,
    Qwen35MoeText,
    Afmoe,
    NemotronHPuzzle,
    ChatGlm,
    Laguna,
    LongcatFlashNgram,
    MuseGlimmerText,
    Plamo3,
    Gemma3nText,
    HrmText,
    InklingText,
    DeepseekV4,
    Glm5NextText,
    HyV4,
    MinimaxM3VlText,
    Dots3Note,
}

impl Family {
    /// The family a config's `model_type` names, if Ster implements it.
    pub(in crate::runtime::load) fn of(model_type: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|family| family.model_type() == model_type)
    }

    /// The family a config without `model_type` belongs to, from the model
    /// class its `architectures` names, for the families whose published
    /// configs leave `model_type` to their remote-code config class.
    pub(in crate::runtime::load) fn of_architectures(raw: &Value) -> Option<Self> {
        let names = raw.get("architectures")?.as_array()?;
        Self::ALL.iter().copied().find(|family| {
            family
                .remote_class()
                .is_some_and(|class| names.iter().any(|name| name.as_str() == Some(class)))
        })
    }

    /// The remote-code model class that stands in for `model_type` in this
    /// family's configs, where they omit it.
    fn remote_class(self) -> Option<&'static str> {
        match self {
            Self::LongcatFlashNgram => Some("LongcatFlashNgramForCausalLM"),
            Self::MinimaxM3VlText => Some("MiniMaxM3SparseForCausalLM"),
            _ => None,
        }
    }

    /// Whether the family's Transformers config class leaves
    /// `tie_word_embeddings` at the library default, true, so its configs
    /// often omit the key.
    pub(in crate::runtime::load) fn tied_by_default(self) -> bool {
        matches!(
            self,
            Self::Gemma
                | Self::Gemma2
                | Self::Gemma3Text
                | Self::Gemma3nText
                | Self::Gemma4Text
                | Self::Gemma4UnifiedText
                | Self::VaultGemma
                | Self::Cohere
                | Self::Cohere2
                | Self::Cohere2Moe
                | Self::Starcoder2
                | Self::Ernie45
                | Self::Ernie45Moe
                | Self::Gpt2
                | Self::GptBigCode
                | Self::Opt
                | Self::Bloom
                | Self::Falcon
                | Self::Mpt
                | Self::Mamba
                | Self::FalconMamba
        )
    }
}
