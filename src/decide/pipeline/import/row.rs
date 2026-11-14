//! One line of a JSONL export read into a labelled choice, or the declared
//! cause it is refused under (`ster.decisions.import.rows`,
//! src/failures/decide/rows.rs), with the values that cause names.

use std::collections::BTreeMap;

use serde_json::{Map, Value};
use wisent_errors::Cause;

use crate::decide::MAX_OPTIONS;
use crate::failures::{Rejection, decide::rows as causes, excerpt, shape};

/// One row as it becomes an example: its context, its options trimmed, and
/// the index of the right one.
pub(super) struct Row {
    pub context: String,
    pub options: Vec<String>,
    pub label: usize,
}

/// The refusal of a line for `cause`, with the values its sentences name
/// beside the line and the path, which the caller adds, and the row's shape
/// and example its fix shows.
fn refused<T>(
    cause: &'static Cause,
    mut values: Vec<(&'static str, String)>,
) -> Result<T, Rejection> {
    values.push(("row_fields", causes::ROW_FIELDS.to_owned()));
    values.push(("row_example", causes::ROW_EXAMPLE.to_owned()));
    Err(Rejection::new(cause, values))
}

/// The field `name` of the row, or the refusal of a row without it.
fn field<'a>(fields: &'a Map<String, Value>, name: &str) -> Result<&'a Value, Rejection> {
    match fields.get(name) {
        Some(value) => Ok(value),
        None => {
            let held: Vec<&str> = fields.keys().map(String::as_str).collect();
            let held = if held.is_empty() { "none".to_owned() } else { excerpt(&held.join(", ")) };
            refused(&causes::FIELD_MISSING, vec![("field", name.to_owned()), ("fields", held)])
        }
    }
}

/// `read`, one line wisent-formats' JSON Lines reader read on its own, as one
/// `{"context", "options", "label"}` row.
pub(super) fn read(given: wisent_formats::jsonl::Line<'_>) -> Result<Row, Rejection> {
    let line = given.text;
    let value: Value = match given.value {
        Ok(value) => value,
        Err(unread) => {
            return refused(
                &causes::NOT_JSON,
                vec![
                    ("excerpt", excerpt(line)),
                    ("column", unread.column.to_string()),
                    ("error", unread.said),
                ],
            );
        }
    };
    let Value::Object(fields) = &value else {
        return refused(
            &causes::NOT_OBJECT,
            vec![("found", shape(Some(&value))), ("excerpt", excerpt(line))],
        );
    };
    let context = match field(fields, "context")? {
        Value::String(text) => text,
        other => {
            return refused(
                &causes::CONTEXT_NOT_TEXT,
                vec![("found", shape(Some(other))), ("excerpt", excerpt(line))],
            );
        }
    };
    let stated = field(fields, "options")?;
    let Value::Array(items) = stated else {
        return refused(
            &causes::OPTIONS_NOT_LIST,
            vec![("found", shape(Some(stated))), ("excerpt", excerpt(line))],
        );
    };
    let shown = excerpt(&stated.to_string());
    let mut options = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        match item.as_str() {
            Some(text) => options.push(text.trim().to_owned()),
            None => {
                return refused(
                    &causes::OPTION_NOT_TEXT,
                    vec![
                        ("index", index.to_string()),
                        ("found", shape(Some(item))),
                        ("options", shown),
                    ],
                );
            }
        }
    }
    let label = field(fields, "label")?;
    let Some(label) = label.as_u64() else {
        return refused(
            &causes::LABEL_NOT_INDEX,
            vec![("found", shape(Some(label))), ("options", shown)],
        );
    };
    let label = checked(context, &options, label, shown)?;
    Ok(Row {
        context: context.clone(),
        options,
        label,
    })
}

/// The checks a well-formed row still fails, in the order a reader fixes
/// them: the context, the number of options, each option, then the label.
fn checked(context: &str, options: &[String], label: u64, shown: String) -> Result<usize, Rejection> {
    if context.trim().is_empty() {
        return refused(&causes::CONTEXT_BLANK, vec![("context", format!("{context:?}"))]);
    }
    let count = options.len().to_string();
    if options.len() < 2 {
        return refused(&causes::OPTIONS_FEW, vec![("count", count), ("options", shown)]);
    }
    if options.len() > MAX_OPTIONS {
        return refused(
            &causes::OPTIONS_MANY,
            vec![("count", count), ("max", MAX_OPTIONS.to_string()), ("options", shown)],
        );
    }
    if let Some(index) = options.iter().position(String::is_empty) {
        return refused(
            &causes::OPTION_BLANK,
            vec![("index", index.to_string()), ("options", shown)],
        );
    }
    let mut seen = BTreeMap::new();
    for (second, option) in options.iter().enumerate() {
        if let Some(first) = seen.insert(option.as_str(), second) {
            return refused(
                &causes::OPTIONS_REPEATED,
                vec![
                    ("first", first.to_string()),
                    ("second", second.to_string()),
                    ("text", format!("{option:?}")),
                    ("options", shown),
                ],
            );
        }
    }
    // A label too large for this machine's indices is past the options too.
    match usize::try_from(label) {
        Ok(index) if index < options.len() => Ok(index),
        _ => refused(
            &causes::LABEL_OUTSIDE,
            vec![
                ("label", label.to_string()),
                ("count", count),
                ("last", (options.len() - 1).to_string()),
                ("options", shown),
            ],
        ),
    }
}
