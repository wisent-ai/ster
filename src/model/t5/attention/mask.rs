//! Padding is measured by row lengths, never by token value: T5's decoder
//! start token may also be its padding token and must remain visible.

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};

pub(in crate::model::t5) fn padding(
    lengths: &[usize],
    sequence: usize,
    device: &Device,
) -> Result<Option<Tensor>> {
    validate_lengths(lengths, sequence)?;
    if lengths.iter().all(|length| *length == sequence) {
        return Ok(None);
    }
    let cells = lengths
        .len()
        .checked_mul(sequence)
        .context("T5 encoder padding-mask shape overflow")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(cells)
        .context("cannot allocate T5 encoder padding mask")?;
    for length in lengths {
        // Positions enumerate the source sequence; a masked key is a boolean byte.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        values.extend((0..sequence).map(|position| u8::from(position >= *length)));
    }
    // One key mask broadcasts over both heads and query positions.
    // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
    Ok(Some(Tensor::from_vec(
        values,
        (lengths.len(), 1, 1, sequence),
        device,
    )?))
}

pub(in crate::model::t5) fn causal(
    lengths: Option<&[usize]>,
    queries: usize,
    start: usize,
    device: &Device,
) -> Result<Option<Tensor>> {
    if let Some(lengths) = lengths {
        validate_lengths(lengths, queries)?;
    }
    // A single appended query sees every cached key and itself.
    // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
    if queries == 1 && lengths.is_none() {
        return Ok(None);
    }
    let keys = start
        .checked_add(queries)
        .context("T5 decoder mask position overflow")?;
    // Without per-row padding, the causal mask is shared across the batch.
    // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
    let batch = match lengths {
        Some(lengths) => lengths.len(),
        None => 1,
    };
    let cells = batch
        .checked_mul(queries)
        .and_then(|cells| cells.checked_mul(keys))
        .context("T5 decoder causal-mask shape overflow")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(cells)
        .context("cannot allocate T5 decoder causal mask")?;
    // Each row has its own valid keys; query positions include the cached prefix.
    // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
    for row in 0..batch {
        let valid_keys = match lengths {
            Some(lengths) => start + lengths[row],
            None => keys,
        };
        for query in start..keys {
            // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
            values.extend((0..keys).map(|key| u8::from(key > query || key >= valid_keys)));
        }
    }
    // The mask's head axis broadcasts over every attention head.
    // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
    Ok(Some(Tensor::from_vec(
        values,
        (batch, 1, queries, keys),
        device,
    )?))
}

fn validate_lengths(lengths: &[usize], width: usize) -> Result<()> {
    if lengths.is_empty() {
        bail!("T5 padding lengths contain no rows");
    }
    for (row, length) in lengths.iter().enumerate() {
        // An empty row has no visible key for the attention softmax.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        if *length == 0 || *length > width {
            bail!("T5 row {row} has valid length {length} outside its tensor width {width}");
        }
    }
    Ok(())
}
