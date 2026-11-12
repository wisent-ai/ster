//! The first-use journey compiled into this binary: who it is for, its
//! screens, and every transition's target and priority. It is embedded at
//! build time, so it is always the file this code was written with; what the
//! reader relies on is checked when it is read, and a journey that breaks it
//! is a defect of the build that every onboarding run, its test first, meets.

use serde::Deserialize;

use super::{FIRST_SUCCESS_FACT, JOURNEY_ID, JOURNEY_VERSION, PRODUCT_ID};

const DEFINITION: &str = include_str!("first_use.json");

#[derive(Debug, Deserialize)]
pub(super) struct Journey {
    product_id: String,
    journey_id: String,
    journey_version: String,
    first_success_fact: String,
    pub(super) entry_screen_id: String,
    pub(super) screens: Vec<Screen>,
}

#[derive(Debug, Deserialize)]
pub(super) struct Screen {
    pub(super) screen_id: String,
    pub(super) presentation: Presentation,
    #[serde(default)]
    transitions: Vec<Transition>,
    #[serde(default)]
    completion_evidence: Option<Evidence>,
}

#[derive(Debug, Deserialize)]
pub(super) struct Presentation {
    pub(super) title: String,
    pub(super) body: String,
}

#[derive(Debug, Deserialize)]
struct Transition {
    next_screen_id: String,
    priority: u64,
}

#[derive(Debug, Deserialize)]
struct Evidence {
    fact: String,
}

/// The embedded journey, read and checked against what the walkthrough
/// relies on: its identity, an entry screen it holds, unique screen ids,
/// every transition leading to a screen it holds at a priority unique on its
/// screen, and the screen whose evidence is the first success.
pub(super) fn journey() -> Journey {
    let journey: Journey = serde_json::from_str(DEFINITION)
        .expect("first_use.json is the journey this reader was written for");
    assert_eq!(
        (
            journey.product_id.as_str(),
            journey.journey_id.as_str(),
            journey.journey_version.as_str(),
            journey.first_success_fact.as_str(),
        ),
        (PRODUCT_ID, JOURNEY_ID, JOURNEY_VERSION, FIRST_SUCCESS_FACT),
        "first_use.json is the journey this binary names"
    );
    let ids: Vec<&str> = journey.screens.iter().map(|screen| screen.screen_id.as_str()).collect();
    assert!(ids.contains(&journey.entry_screen_id.as_str()), "the entry screen is a screen");
    for (index, screen) in journey.screens.iter().enumerate() {
        assert!(!ids[..index].contains(&screen.screen_id.as_str()), "screen ids are unique");
        let mut priorities = std::collections::HashSet::new();
        for transition in &screen.transitions {
            assert!(ids.contains(&transition.next_screen_id.as_str()), "transitions lead to screens");
            assert!(priorities.insert(transition.priority), "priorities are unique per screen");
        }
    }
    assert!(journey.first_success().is_some(), "one screen is the first success");
    journey
}

impl Journey {
    /// The screen `id` names, when the journey holds one.
    pub(super) fn screen(&self, id: &str) -> Option<&Screen> {
        self.screens.iter().find(|screen| screen.screen_id == id)
    }

    /// The ids of every screen, as a refusal lists them.
    pub(super) fn screen_ids(&self) -> String {
        let ids: Vec<&str> = self.screens.iter().map(|screen| screen.screen_id.as_str()).collect();
        ids.join(", ")
    }

    /// The screen whose completion evidence is the first success.
    pub(super) fn first_success(&self) -> Option<&Screen> {
        self.screens.iter().find(|screen| {
            screen
                .completion_evidence
                .as_ref()
                .is_some_and(|evidence| evidence.fact == FIRST_SUCCESS_FACT)
        })
    }
}

impl Screen {
    /// The target of the transition that leads: the lowest priority, as the
    /// journey contract and Echo's own routing order transitions.
    pub(super) fn next(&self) -> Option<&str> {
        self.transitions
            .iter()
            .min_by_key(|transition| transition.priority)
            .map(|transition| transition.next_screen_id.as_str())
    }
}
