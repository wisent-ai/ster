//! Refusal detection for synthesised negatives.
//!
//! A port of Wisent's `BaseRefusaler`. Synthesis asks a model to produce a
//! response that *lacks* a trait; safety-tuned open-weight models answer that
//! request with "I can't help with that" surprisingly often, and a refusal is a
//! useless negative because it differs from the positive along the refusal axis
//! rather than along the trait axis. Detecting refusals cheaply and lexically —
//! no second model call — keeps the synthesis loop fast enough to be interactive.

use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;
use unicode_normalization::UnicodeNormalization;

/// Score at or above which a text is treated as a refusal.
pub const DEFAULT_THRESHOLD: f32 = 0.5;

/// A family of refusal phrasing, one per alternation branch of the pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    AiDisclaimer,
    Policy,
    ApologyHedge,
    Unable,
    CannotAction,
    PreferRather,
    DeclineRefuse,
    NoSupport,
    NoAbility,
    RefusalWord,
}

impl Family {
    /// Confidence that a hit in this family really is a refusal.
    ///
    /// Verbatim `_FAMILY_WEIGHTS` from the Python. "I cannot help" is the only
    /// phrasing that is a refusal on its own (1.0); a bare "sorry" is weak (0.4)
    /// because it prefixes plenty of non-refusals.
    pub fn weight(self) -> f32 {
        match self {
            Self::AiDisclaimer => 0.9,
            Self::Policy => 0.9,
            Self::ApologyHedge => 0.4,
            Self::Unable => 0.9,
            Self::CannotAction => 1.0,
            Self::PreferRather => 0.6,
            Self::DeclineRefuse => 0.9,
            Self::NoSupport => 0.8,
            Self::NoAbility => 0.8,
            Self::RefusalWord => 0.6,
        }
    }

    /// The capture-group name, which is also the wire name in reports.
    pub fn name(self) -> &'static str {
        match self {
            Self::AiDisclaimer => "ai_disclaimer",
            Self::Policy => "policy",
            Self::ApologyHedge => "apology_hedge",
            Self::Unable => "unable",
            Self::CannotAction => "cannot_action",
            Self::PreferRather => "prefer_rather",
            Self::DeclineRefuse => "decline_refuse",
            Self::NoSupport => "no_support",
            Self::NoAbility => "no_ability",
            Self::RefusalWord => "refusal_word",
        }
    }
}

/// Every family, in the alternation order of the pattern.
const FAMILIES: [Family; 10] = [
    Family::AiDisclaimer,
    Family::Policy,
    Family::ApologyHedge,
    Family::Unable,
    Family::CannotAction,
    Family::PreferRather,
    Family::DeclineRefuse,
    Family::NoSupport,
    Family::NoAbility,
    Family::RefusalWord,
];

/// The verdict for one text.
#[derive(Debug, Clone, Serialize)]
pub struct Score {
    pub score: f32,
    pub family: Option<Family>,
    /// The matched text, so an operator reading a report can see *why*.
    pub snippet: String,
}

/// Up to two short filler words between the negation and the action verb, so
/// "i cannot really help" hits the same branch as "i cannot help".
const FILLER: &str = r"(?:\b\w{1,15}\b\s+){0,2}";
const NEG_MODAL: &str =
    r"(?:can(?:\s*not)?|can't|won't|will\s+not|should(?:\s*not)?|shouldn't|must(?:\s*not)?|mustn't)";
const AM: &str = r"(?:i\s+(?:am|['']m))";
const I: &str = r"\bi\b";
const I_M: &str = r"(?:i['']m)";
const VERB_ACTION: &str = concat!(
    r"(?:help|assist|provide|share|give|supply|offer|",
    r"comply(?:\s+with)?|fulfil?l|answer|respond|",
    r"create|generate|write|compose|produce|draft|",
    r"explain|instruct|guide|walk\s+you\s+through|",
    r"facilitate|bypass|circumvent|enable|",
    r"do\s+(?:that|this|so))",
);
const SUPPORT_VERBS: &str = r"(?:support|condone|endorse|promote|encourage)";
const ABILITY_NOUNS: &str = r"(?:ability|capability|capacity|permission|authorization|access)";

/// The ported `_REFUSAL_RE`.
///
/// Assembled from the fragments above rather than written as one `concat!`
/// literal so the fragments stay individually reviewable against the Python.
/// The Python compiles with `re.VERBOSE | re.IGNORECASE | re.UNICODE`; `(?ix)`
/// is the exact equivalent here, since Rust's `regex` is Unicode-aware by
/// default. The pattern uses no backreferences and no lookaround, so it
/// translates one-for-one into the crate's finite-automaton engine, and the
/// crate's leftmost-first alternation semantics match Python's.
///
/// Deviation: the Python appends a dead
/// `.replace("{i_am}", "(?:i\\s+(?:am|['']m))")` to the f-string. No `{i_am}`
/// placeholder survives f-string interpolation, so the replacement never fires
/// and is dropped here.
static REFUSAL_RE: LazyLock<Regex> = LazyLock::new(|| {
    let pattern = [
        "(?ix)",
        "(?:",
        // A) "As an AI (language) model..." disclaimers
        r"(?P<ai_disclaimer>\bas\s+an?\s+(?:ai|assistant)(?:\s+language)?\s+model\b)",
        "|",
        // B) Policy/legal/safety invocations
        "(?P<policy>",
        r"\b(?:this|that|it)\s+would\s+be\s+(?:unethical|unsafe|illegal|harmful)\b",
        "|",
        r"\bagainst\s+(?:policy|the\s+rules|guidelines|terms(?:\s+of\s+service)?)\b",
        "|",
        r"\b(?:due\s+to|for)\s+(?:safety|legal|policy|ethical)\s+(?:reasons|concerns)\b",
        ")",
        "|",
        // C) Apologies / hedges commonly preceding refusals
        "(?P<apology_hedge>",
        r"\b(?:sorry|apologies|unfortunately|regrettably)\b",
        "|",
        r"\b",
        I_M,
        r"\s+afraid\b",
        ")",
        "|",
        // D) Explicit inability / not permitted / not comfortable
        "(?P<unable>",
        AM,
        r"\s+(?:unable|not\s+able|not\s+allowed|not\s+permitted|not\s+comfortable)\s+to\b",
        ")",
        "|",
        // E) "I [negation] [filler] [help/provide/...]"
        "(?P<cannot_action>",
        I,
        r"\s+",
        NEG_MODAL,
        r"\s+",
        FILLER,
        VERB_ACTION,
        r"\b",
        ")",
        "|",
        // F) Prefer/rather not (indirect refusal)
        "(?P<prefer_rather>",
        r"\b(?:i['']d|i\s+would)\s+(?:rather|prefer)\s+not\s+(?:to\s+)?\w+\b",
        ")",
        "|",
        // G) Decline/refuse verbs
        "(?P<decline_refuse>",
        I,
        r"\s+(?:must|have\s+to|will)\s+(?:decline|refuse)\s+to\b",
        "|",
        r"\b(?:i\s+decline|i\s+refuse)\b",
        ")",
        "|",
        // H) "I don't/won't [support|condone|endorse|promote] ..."
        "(?P<no_support>",
        I,
        r"\s+(?:do\s+not|don't|won't|will\s+not)\s+",
        SUPPORT_VERBS,
        r"\b",
        ")",
        "|",
        // I) "I do not have the [ability/capability/permission/access] to ..."
        "(?P<no_ability>",
        I,
        r"\s+(?:do\s+not|don't)\s+have\s+the\s+",
        ABILITY_NOUNS,
        r"\s+to\b",
        ")",
        "|",
        // J) Direct lexical hits
        r"(?P<refusal_word>\brefus(?:e|al)\b)",
        ")",
    ]
    .concat();
    Regex::new(&pattern).expect("refusal pattern is a fixed compile-time constant")
});

/// Scores how strongly `text` reads as a refusal.
///
/// NFKC-normalised and trimmed but deliberately *not* case-folded: the pattern
/// carries `(?i)`, and folding here would only cost an allocation. Multiple
/// families can in principle fire; the highest weight wins, plus a small bonus
/// when an apology hedge co-occurs with a harder signal.
pub fn score(text: &str) -> Score {
    let normalized = text.nfkc().collect::<String>();
    let Some(captures) = REFUSAL_RE.captures(normalized.trim()) else {
        return Score { score: 0.0, family: None, snippet: String::new() };
    };

    let mut best: Option<Family> = None;
    let mut best_weight = 0.0f32;
    for family in FAMILIES {
        if captures.name(family.name()).is_none() {
            continue;
        }
        if family.weight() > best_weight {
            best_weight = family.weight();
            best = Some(family);
        }
    }

    // Ported verbatim, including the fact that it is unreachable: the branches
    // are one flat alternation, so exactly one named group can be set per match
    // and `apology_hedge` never co-occurs with another family. Keeping it means
    // a future non-exclusive branch inherits the intended behaviour.
    let bonus = if captures.name(Family::ApologyHedge.name()).is_some()
        && FAMILIES
            .iter()
            .any(|family| *family != Family::ApologyHedge && captures.name(family.name()).is_some())
    {
        0.1
    } else {
        0.0
    };

    Score {
        score: (best_weight + bonus).min(1.0),
        family: best,
        snippet: captures.get(0).map(|m| m.as_str().to_string()).unwrap_or_default(),
    }
}

/// Whether `text` scores at or above `threshold`.
pub fn looks_like_refusal(text: &str, threshold: f32) -> bool {
    score(text).score >= threshold
}
