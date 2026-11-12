//! One page of the datasets server's rows route: asking for it, reading it,
//! and the class names its feature list gives the label column.

use anyhow::Result;
use serde_json::Value;

use super::{FetchOptions, ROWS_URL};
use crate::{
    decide::MAX_OPTIONS,
    failures::{decide::fetch as causes, refuse},
};

/// The field names of a JSON object, as a refusal lists them.
pub(super) fn fields(value: &Value) -> String {
    match value.as_object() {
        Some(fields) => fields.keys().cloned().collect::<Vec<_>>().join(", "),
        None => format!("none: it is {}", wisent_errors::trim_detail(&value.to_string(), 60)),
    }
}

pub(super) fn read_page(
    agent: &ureq::Agent,
    options: &FetchOptions,
    offset: usize,
    length: usize,
) -> Result<Value> {
    let request = agent
        .get(ROWS_URL)
        .query("dataset", &options.dataset)
        .query("config", &options.config)
        .query("split", &options.split)
        .query("offset", &offset.to_string())
        .query("length", &length.to_string());
    // The request as a refusal names it: the route and every query the page
    // asked with.
    let url = format!(
        "{ROWS_URL}?dataset={}&config={}&split={}&offset={offset}&length={length}",
        options.dataset, options.config, options.split
    );
    let response = stado_wait::blocking(
        stado_wait::Kind::Network,
        format!("rows {offset}..{} of {}", offset + length, options.dataset),
        ROWS_URL,
        || request.call(),
    );
    let dataset = options.dataset.as_str();
    let unread = |status: &str, error: std::io::Error| {
        refuse(
            &causes::PAGE_UNREAD,
            &[
                ("dataset", dataset),
                ("url", &url),
                ("status", status),
                ("kind", &crate::failures::file::kind(&error)),
                ("error", &error.to_string()),
            ],
        )
    };
    let response = match response {
        Ok(response) => response,
        Err(ureq::Error::Status(status, response)) => {
            let code = status.to_string();
            let body = response.into_string().map_err(|error| unread(&code, error))?;
            // The server words its refusal in an `error` field; any other
            // body is shown as it was sent.
            let said = match serde_json::from_str::<Value>(&body) {
                Ok(Value::Object(fields)) => match fields.get("error") {
                    Some(Value::String(error)) => error.clone(),
                    _ => body,
                },
                _ => body,
            };
            let cause = match status {
                429 => &causes::SERVER_LIMITED,
                500..=599 => &causes::SERVER_FAILING,
                _ => &causes::SERVER_REFUSED,
            };
            return Err(refuse(
                cause,
                &[
                    ("dataset", dataset),
                    ("config", &options.config),
                    ("split", &options.split),
                    ("url", &url),
                    ("status", &code),
                    ("said", &wisent_errors::trim_detail(&said, 240)),
                ],
            ));
        }
        Err(ureq::Error::Transport(transport)) => {
            return Err(refuse(
                &causes::SERVER_UNREACHABLE,
                &[
                    ("dataset", dataset),
                    ("url", &url),
                    ("kind", &format!("{:?}", transport.kind())),
                    ("error", &transport.to_string()),
                ],
            ));
        }
    };
    let text = response.into_string().map_err(|error| unread("200", error))?;
    serde_json::from_str(&text).map_err(|error| {
        refuse(
            &causes::PAGE_NOT_JSON,
            &[
                ("dataset", dataset),
                ("url", &url),
                ("page", &wisent_errors::trim_detail(&text, 240)),
                ("size", &text.len().to_string()),
                ("line", &error.line().to_string()),
                ("column", &error.column().to_string()),
                ("error", &error.to_string()),
            ],
        )
    })
}

/// The class names of the label column, from the page's feature list.
pub(super) fn class_names(page: &Value, options: &FetchOptions) -> Result<Vec<String>> {
    let label = options.label_field.as_str();
    let dataset = options.dataset.as_str();
    let Some(features) = page["features"].as_array() else {
        return Err(refuse(
            &causes::PAGE_FEATURELESS,
            &[("dataset", dataset), ("fields", &fields(page))],
        ));
    };
    let Some(feature) = features.iter().find(|feature| feature["name"].as_str() == Some(label))
    else {
        let columns: Vec<&str> = features.iter().filter_map(|feature| feature["name"].as_str()).collect();
        return Err(refuse(
            &causes::LABEL_ABSENT,
            &[("dataset", dataset), ("label", label), ("columns", &columns.join(", "))],
        ));
    };
    let Some(listed) = feature["type"]["names"].as_array() else {
        return Err(refuse(
            &causes::LABEL_NOT_CLASS,
            &[
                ("dataset", dataset),
                ("label", label),
                ("feature", &wisent_errors::trim_detail(&feature["type"].to_string(), 120)),
            ],
        ));
    };
    let mut names = Vec::with_capacity(listed.len());
    for (index, name) in listed.iter().enumerate() {
        let Some(name) = name.as_str() else {
            return Err(refuse(
                &causes::CLASS_NOT_TEXT,
                &[
                    ("dataset", dataset),
                    ("label", label),
                    ("index", &index.to_string()),
                    ("found", &name.to_string()),
                ],
            ));
        };
        names.push(name.to_owned());
    }
    let values = [
        ("dataset", dataset),
        ("label", label),
        ("count", &names.len().to_string()),
        ("most", &MAX_OPTIONS.to_string()),
        ("classes", &names.join(", ")),
    ];
    if names.len() < 2 {
        return Err(refuse(&causes::CLASSES_TOO_FEW, &values));
    }
    if names.len() > MAX_OPTIONS {
        return Err(refuse(&causes::CLASSES_TOO_MANY, &values));
    }
    Ok(names)
}
