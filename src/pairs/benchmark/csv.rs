//! The CSV the benchmark exports use: RFC 4180 records, comma separated,
//! fields optionally quoted, a quote inside a quoted field doubled, and line
//! breaks allowed inside quotes. The first record names the columns.

use std::collections::BTreeMap;

use anyhow::{bail, Result};

/// One record keyed by its header name, with its one-based record number.
pub(super) struct Record {
    pub number: usize,
    pub fields: BTreeMap<String, String>,
}

impl Record {
    pub fn get(&self, column: &str) -> &str {
        self.fields
            .get(column)
            .map(|value| value.trim())
            .unwrap_or_default()
    }
}

pub(super) fn records(text: &str, label: &str) -> Result<Vec<Record>> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut rows = rows(text, label)?.into_iter();
    let Some(header) = rows.next() else {
        bail!("{label} holds no header row");
    };
    let header: Vec<String> = header
        .into_iter()
        .map(|name| name.trim().to_string())
        .collect();
    let mut records = Vec::new();
    for (index, row) in rows.enumerate() {
        if row.len() == 1 && row[0].is_empty() {
            continue;
        }
        if row.len() != header.len() {
            bail!(
                "{label} record {} has {} fields where the header names {}",
                index + 1,
                row.len(),
                header.len()
            );
        }
        records.push(Record {
            number: index + 1,
            fields: header.iter().cloned().zip(row).collect(),
        });
    }
    Ok(records)
}

fn rows(text: &str, label: &str) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        match (quoted, character) {
            (true, '"') if characters.peek() == Some(&'"') => {
                characters.next();
                field.push('"');
            }
            (true, '"') => quoted = false,
            (true, other) => field.push(other),
            (false, '"') if field.is_empty() => quoted = true,
            (false, ',') => row.push(std::mem::take(&mut field)),
            (false, '\r') if characters.peek() == Some(&'\n') => {}
            (false, '\n') => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            (false, other) => field.push(other),
        }
    }
    if quoted {
        bail!("{label} ends inside a quoted field");
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}
