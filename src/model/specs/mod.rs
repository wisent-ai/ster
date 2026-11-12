//! The parts an [`Architecture`](super::Architecture) is assembled from, one
//! type per way a family departs from the plain Llama block.

mod attention;
mod block;
#[path = "experts_kept/experts.rs"]
mod experts;
mod families;
#[path = "experts_kept/mixers.rs"]
mod mixers;
mod schedule;
#[path = "experts_kept/sparse.rs"]
mod sparse;

pub use attention::{
    GateFunction, GlobalAttention, LatentAttention, NgramSpec, PerLayerInputSpec, Positions,
    QkvLayout, WindowLatent,
};
pub use block::{Activation, FeedForwardKind, NormKind, QueryKeyNorm, RopeScaling};
pub use experts::{
    DenseClamp, ExpertGroups, ExpertLayout, LatentExperts, MixtureOfExperts, QueryTemperature,
    Scoring, SharedExpert, SharedForm, SwigluLimit,
};
pub use families::{
    CompressYarn, CompressedSpec, HyperConnections, HyperForm, InklingSpec, RelativeHeads,
};
pub use mixers::{
    DeltaRuleForm, DeltaRuleSpec, LightningForm, LightningSpec, ParallelScan, ParameterNorm,
    ScaledResiduals, SharedBlocksSpec, ShortConvolution, StateSpaceSpec, StructuredSpec,
};
pub use schedule::{LayerPlan, Loops, Recurrence, SkipConnections};
pub use sparse::{BlockIndexSpec, IndexerSpec, KeyPool};
