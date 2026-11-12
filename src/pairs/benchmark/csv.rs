//! The CSV the benchmark exports use: RFC 4180 records, comma separated,
//! fields optionally quoted, a quote inside a quoted field doubled, and line
//! breaks allowed inside quotes. The first record names the columns; a
//! leading byte-order mark is not part of the first column's name.

use std::collections::BTreeMap;

use anyhow::Result;

use crate::failures::{pairs::export, refuse};

/// One record keyed by its header name, with its one-based record number.
pub(super) struct Record {
    pub number: usize,
    pub fields: BTreeMap<String, String>,
}

impl Record {
    /// The trimmed value of `column`, one of the columns [`records`] was
    /// told the export names, so every record holds it.
    pub fn get(&self, column: &str) -> &str {
        self.fields.get(column).map_or("", |value| value.trim())
    }
}

/// Names, quoted and joined, as a refusal shows them.
fn listed<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    names
        .into_iter()
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The records of the `benchmark` export at `path`, whose header must name
/// every column of `required`: a file without one is another export, not
/// one whose every row lacks the value.
pub(super) fn records(
    text: &str,
    path: &str,
    benchmark: &str,
    required: &[&str],
) -> Result<Vec<Record>> {
    let wanted = listed(required.iter().copied());
    let mut rows = rows(text.trim_start_matches('\u{feff}'), path, benchmark)?.into_iter();
    let Some(header) = rows.next() else {
        return Err(refuse(
            &export::CSV_NO_HEADER,
            &[
                ("path", path),
                ("benchmark", benchmark),
                ("required", &wanted),
                ("bytes", &text.len().to_string()),
            ],
        ));
    };
    let header: Vec<String> = header
        .into_iter()
        .map(|name| name.trim().to_string())
        .collect();
    let named = listed(header.iter().map(String::as_str));
    if let Some(column) = required
        .iter()
        .find(|column| !header.iter().any(|name| name == *column))
    {
        return Err(refuse(
            &export::CSV_COLUMN_MISSING,
            &[
                ("path", path),
                ("column", column),
                ("benchmark", benchmark),
                ("columns", &named),
                ("required", &wanted),
            ],
        ));
    }
    let mut records = Vec::new();
    for (number, row) in (super::choices::FIRST_ROW..).zip(rows) {
        // A blank line is one empty field: it holds no record.
        if matches!(row.as_slice(), [only] if only.is_empty()) {
            continue;
        }
        if row.len() != header.len() {
            return Err(refuse(
                &export::CSV_RECORD_WIDTH,
                &[
                    ("path", path),
                    ("number", &number.to_string()),
                    ("fields", &row.len().to_string()),
                    ("width", &header.len().to_string()),
                    ("header", &named),
                    ("benchmark", benchmark),
                ],
            ));
        }
        records.push(Record {
            number,
            fields: header.iter().cloned().zip(row).collect(),
        });
    }
    Ok(records)
}

fn rows(text: &str, path: &str, benchmark: &str) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    // The line being read and the line the open quote was opened on, both
    // counted from one, so a refusal names where to look.
    let mut line = 1usize;
    let mut opened = 0usize;
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '\n' {
            line += 1;
        }
        match (quoted, character) {
            (true, '"') if characters.peek() == Some(&'"') => {
                characters.next();
                field.push('"');
            }
            (true, '"') => quoted = false,
            (true, other) => field.push(other),
            (false, '"') if field.is_empty() => {
                quoted = true;
                opened = line;
            }
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
        let start: String = field.chars().take(40).collect();
        return Err(refuse(
            &export::CSV_QUOTE_OPEN,
            &[
                ("path", path),
                ("benchmark", benchmark),
                ("line", &opened.to_string()),
                ("lines", &line.to_string()),
                ("record", &(rows.len() + 1).to_string()),
                ("start", &start),
            ],
        ));
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}
