//! Which layers run, in what order and how often: DeciLM's per-layer plans,
//! loops, HRM-Text's recurrence and Solar's skip connections.

use candle_core::Tensor;

/// One DeciLM layer (`block_configs`): its key-value heads, `None` for a
/// no-op attention, and its feed-forward width, `None` for a no-op
/// feed-forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerPlan {
    pub key_value_heads: Option<usize>,
    pub intermediate: Option<usize>,
}

/// A looped decoder: `physical` stored layers run in order as many times as
/// the config's layer count holds them, each pass with its own key-value
/// cache slots. Under `norm_between` (Nanbeige's `num_loops`) the final
/// norm also closes every pass but the last; with `gate_window`
/// (IQuest-LoopCoder's `loop_window_size`) every later pass mixes global
/// attention over the first pass's keys with local attention over its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Loops {
    pub physical: usize,
    pub count: usize,
    pub norm_between: bool,
    pub gate_window: Option<usize>,
}

/// HRM-Text's schedule: a low stack and a high stack of `per_stack` stored
/// layers each, run `high_cycles` times as `low_cycles` passes of the low
/// stack and one of the high stack. Every pass is `per_stack` Ster layers
/// with their own key-value slots. A low pass reads the low state plus the
/// high state and replaces the low state; a high pass reads the high state
/// plus the low state and replaces the high state; each pass ends in a
/// scale-free norm. The high state starts as the scaled embedding, the low
/// state as `z_L_init`, and the last high state is the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recurrence {
    pub per_stack: usize,
    pub high_cycles: usize,
    pub low_cycles: usize,
}

impl Recurrence {
    /// How many Ster layers the schedule runs.
    pub fn layers(self) -> usize {
        self.high_cycles * (self.low_cycles + 1) * self.per_stack
    }

    /// Whether Ster layer `layer` belongs to a low pass.
    pub fn low(self, layer: usize) -> bool {
        (layer / self.per_stack) % (self.low_cycles + 1) < self.low_cycles
    }

    /// The stored layer Ster layer `layer` runs: the low stack's first, then
    /// the high stack's.
    pub fn stored(self, layer: usize) -> usize {
        let within = layer % self.per_stack;
        if self.low(layer) {
            within
        } else {
            self.per_stack + within
        }
    }
}

/// Solar's block skip connections (`bskcn_1` to `bskcn_4`): before the
/// layers in `save[i]` the hidden state is kept in slot `i`; before the
/// layers in `blend[i]` it becomes `weight · kept[i] + (1 − weight) · hidden`,
/// `weight` being the inference value of `bskcn_tv`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkipConnections {
    pub save: [u128; 2],
    pub blend: [u128; 2],
    pub weight: f64,
}

impl SkipConnections {
    /// The hidden state layer `layer` reads, keeping what it must in `kept`.
    pub(crate) fn apply(
        &self,
        layer: usize,
        hidden: Tensor,
        kept: &mut [Option<Tensor>; 2],
    ) -> candle_core::Result<Tensor> {
        let listed = |set: u128| layer < u128::BITS as usize && set & (1u128 << layer) != 0;
        for slot in 0..2 {
            if listed(self.save[slot]) {
                kept[slot] = Some(hidden.clone());
            }
        }
        let mut hidden = hidden;
        // The reading holds every slot's first save at or before its first
        // blend, so a blending layer finds the slot filled.
        for slot in 0..2 {
            if let (true, Some(earlier)) = (listed(self.blend[slot]), &kept[slot]) {
                hidden = ((earlier * self.weight)? + (hidden * (1.0 - self.weight))?)?;
            }
        }
        Ok(hidden)
    }
}
