//! Every area page under docs/guide/errors/ is, below its generated-section
//! marker, byte for byte what `Operation::markdown` renders for the
//! functionalities of Ster's catalogue that name it, in declaration order.
//! A page edited by hand, a cause changed without its page, or a page no
//! functionality names fails here, so the published catalogue cannot drift
//! from the declarations the refusals are raised from.

use std::path::Path;

use ster::failures::{FAILURE_EXIT, operations};

/// The line every area page's generated section starts below.
const MARKER: &str = "<!-- generated: ster errors list --markdown -->\n";

/// Where a functionality's `docs` address names its page in this checkout.
const BLOB: &str = "https://github.com/wisent-ai/ster/blob/main/";

#[test]
fn every_error_page_is_its_catalogue_rendered() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let catalogue = operations();
    let mut pages: Vec<&str> = Vec::new();
    for operation in &catalogue {
        let page = operation.docs.strip_prefix(BLOB).unwrap_or_else(|| {
            panic!(
                "{} names its page as {}, which is not a page of this repository",
                operation.id, operation.docs
            )
        });
        if !pages.contains(&page) {
            pages.push(page);
        }
    }
    for page in &pages {
        let text = std::fs::read_to_string(root.join(page)).unwrap_or_else(|error| {
            panic!("{page}, which the catalogue names, could not be read: {error}")
        });
        let (_, published) = text
            .split_once(MARKER)
            .unwrap_or_else(|| panic!("{page} holds no generated-section marker {MARKER:?}"));
        let rendered = catalogue
            .iter()
            .filter(|operation| operation.docs.ends_with(&format!("/{page}")))
            .map(|operation| operation.markdown(FAILURE_EXIT))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            published, rendered,
            "{page} below its marker differs from Operation::markdown of its catalogue; replace that section with `ster errors list --markdown --page {page}`"
        );
    }
    let held = std::fs::read_dir(root.join("docs/guide/errors"))
        .unwrap_or_else(|error| panic!("docs/guide/errors could not be listed: {error}"));
    for entry in held {
        let name = entry.expect("a docs/guide/errors entry").file_name();
        let page = format!("docs/guide/errors/{}", name.to_string_lossy());
        assert!(
            pages.contains(&page.as_str()),
            "{page} is no functionality's page; name it from a functionality's `docs` or remove it"
        );
    }
}
