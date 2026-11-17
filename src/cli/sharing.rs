//! Whether `ster sources` reports each source it saw to Wisent Integrations,
//! and what it sends: the onboarding client's [`SourcesSharing`], declared once.

use wisent_onboarding_client::SourcesSharing;

pub(crate) const SHARING: SourcesSharing =
    SourcesSharing::new("ster", "STER_STADO_INTEGRATION_TOKEN", "ster sources share yes | no", "IMP-STER-06");
