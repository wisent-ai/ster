//! The text the model reads for one question: the state, the question, and
//! the options under answer letters. One rendering per option order, because
//! a language model prefers some letters over others regardless of what they
//! label, and showing every option under every letter is what cancels that.

use super::request::{text_of, Question, Request};

/// The answer letters, in the order options are listed.
pub const LABELS: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
];

/// The option orders one question is shown in.
///
/// `requested` is the operator's setting: `0` asks for every cyclic shift, so
/// each option is read under each letter exactly once and the letter
/// preference averages out completely; any other value caps the number of
/// shifts, and one shift is a single pass with no correction. There are never
/// more shifts than options, since the shifts repeat after that.
pub fn orders(count: usize, requested: usize) -> Vec<Vec<usize>> {
    let shifts = if requested == 0 { count } else { requested.min(count) };
    (0..shifts)
        .map(|shift| (0..count).map(|position| (position + shift) % count).collect())
        .collect()
}

/// The prompt for one question with its options in `order`: position `j`
/// under letter `j` shows option `order[j]`.
pub fn render(request: &Request, question: &Question, options: &[String], order: &[usize]) -> String {
    let mut prompt = String::with_capacity(512);
    prompt.push_str(
        "Decide the question below about the state. Reply with the letter of the one option that fits best.\n\nState:\n",
    );
    prompt.push_str(text_of(&request.state).trim());
    prompt.push_str("\n\nQuestion: ");
    prompt.push_str(text_of(question.instructions()).trim());
    prompt.push_str("\n\nOptions:\n");
    for (position, &index) in order.iter().enumerate() {
        prompt.push_str(LABELS[position]);
        prompt.push_str(". ");
        prompt.push_str(options[index].trim());
        prompt.push('\n');
    }
    prompt.push_str("\nAnswer:");
    prompt
}
