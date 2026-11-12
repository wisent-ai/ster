//! How auditing a pair set fails, and the near-duplicate settings synthesis
//! shares with it. Every setting is checked before any set is read.

use wisent_errors::{Cause, Code, Operation};

use super::super::workspace::ACTIVE;
use super::{DOCS, FILE};

pub static BANDS_INVALID: Cause = Cause {
    id: "bands-invalid",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "checking the near-duplicate settings",
    cause: "dedupe band count {bands} must divide the 64-bit simhash evenly, such as 4, 8, 16, or 32",
    observed: "--dedupe-bands {bands}",
    remedy: "give --dedupe-bands a divisor of 64: 1, 2, 4, 8, 16, 32 or 64",
    test: "tests/pairs/imports/inspect.sh::a_band_count_that_does_not_divide_64_is_refused",
};

pub static THRESHOLD_TOO_WIDE: Cause = Cause {
    id: "threshold-too-wide",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "checking the near-duplicate settings",
    cause: "dedupe threshold {bits} exceeds the 64-bit simhash width",
    observed: "--dedupe-bits {bits}",
    remedy: "give --dedupe-bits a Hamming distance from 0 to 64",
    test: "tests/pairs/imports/inspect.sh::a_threshold_wider_than_the_fingerprint_is_refused",
};

pub static RATIO_INVALID: Cause = Cause {
    id: "ratio-invalid",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "checking the length balance setting",
    cause: "a pair's longer side over its shorter side is 1 or more, so a ratio below 1, or one that is not a number, counts every pair or none as unbalanced",
    observed: "--unbalanced-ratio {ratio}",
    remedy: "give --unbalanced-ratio a number of 1 or more: 3 counts a pair whose longer side has over three times the characters of the other",
    test: "tests/pairs/imports/inspect.sh::a_ratio_below_one_is_refused",
};

/// The near-duplicate settings `inspect` and `synthesize` take.
pub static DEDUPE: Operation = Operation {
    id: "ster.pairs.dedupe",
    entries: &[],
    uses: &[],
    summary: "check the near-duplicate settings a caller gives: the SimHash Hamming threshold and the band count",
    docs: DOCS,
    warnings: &[],
    causes: &[&BANDS_INVALID, &THRESHOLD_TOO_WIDE],
};

/// Auditing one set without a model.
pub static INSPECT: Operation = Operation {
    id: "ster.pairs.inspect",
    entries: &["ster pairs inspect", "ster request pairs/inspect"],
    uses: &[&DEDUPE, &ACTIVE, &FILE],
    summary: "audit a pair set without a model: near-duplicates, the length balance of each pair, and the diversity of its positives",
    docs: DOCS,
    warnings: &[],
    causes: &[&RATIO_INVALID],
};
