//! Where a choice sits among a question's options.

use super::*;

/// The position `answer` names under --answer-form index: a whole number
/// from 0, or text holding one, inside the row's choices.
pub(super) fn index_of(
    answer: &Value,
    pointer: &str,
    choices: &[&str],
    shown: String,
) -> Result<usize, Rejection> {
    let stated = match (answer.as_u64(), answer.as_str()) {
        (Some(index), _) => index,
        (None, Some(text)) => match text.trim().parse::<u64>() {
            Ok(index) => index,
            Err(error) => {
                return refused(
                    &causes::ANSWER_NOT_INDEX,
                    vec![
                        ("pointer", pointer.to_owned()),
                        ("found", shape(Some(answer))),
                        ("detail", format!("the text is not a whole number: {error}")),
                    ],
                );
            }
        },
        (None, None) => {
            return refused(
                &causes::ANSWER_NOT_INDEX,
                vec![
                    ("pointer", pointer.to_owned()),
                    ("found", shape(Some(answer))),
                    ("detail", "it is neither a whole number nor text".to_owned()),
                ],
            );
        }
    };
    // An index too large for this machine is past the choices too.
    match usize::try_from(stated) {
        Ok(index) if index < choices.len() => Ok(index),
        _ => refused(
            &causes::ANSWER_INDEX_OUTSIDE,
            vec![
                ("index", stated.to_string()),
                ("count", choices.len().to_string()),
                ("last", (choices.len() - 1).to_string()),
                ("pointer", pointer.to_owned()),
                ("choices", shown),
            ],
        ),
    }
}
