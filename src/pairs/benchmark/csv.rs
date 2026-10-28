//! The CSV the benchmark exports use: RFC 4180 records, comma separated,
//! fields optionally quoted, a quote inside a quoted field doubled, and line
//! breaks allowed inside quotes. The first record names the columns; a
//! leading byte-order mark is not part of the first column's name.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

/// One record keyed by its header name, with its one-based record number.
pub(super) struct Record {
    pub number: usize,
    pub fields: BTreeMap<String, String>,
}

impl Record {
    /// The trimmed value of `column`. A column the header does not name is
    /// refused with the columns it does name: a file without it is another
    /// export, not one whose every row lacks the value.
    pub fn get(&self, column: &str) -> Result<&str> {
        match self.fields.get(column) {
            Some(value) => Ok(value.trim()),
            None => bail!(
                "record {} has no column {column:?}; the header names {}",
                self.number,
                self.fields.keys().map(|name| format!("{name:?}")).collect::<Vec<_>>().join(", ")
            ),
        }
    }
}

pub(super) fn records(text: &str, label: &str) -> Result<Vec<Record>> {
    let mut rows = rows(text.trim_start_matches('\u{feff}'), label)?.into_iter();
    let Some(header) = rows.next() else {
        bail!("{label} holds no header row");
    };
    let header: Vec<String> = header
        .into_iter()
        .map(|name| name.trim().to_string())
        .collect();
    let mut records = Vec::new();
    for (number, row) in (super::choices::FIRST_ROW..).zip(rows) {
        // A blank line is one empty field: it holds no record.
        if matches!(row.as_slice(), [only] if only.is_empty()) {
            continue;
        }
        if row.len() != header.len() {
            bail!(
                "{label} record {number} has {} fields where the header names {}",
                row.len(),
                header.len()
            );
        }
        records.push(Record {
            number,
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
