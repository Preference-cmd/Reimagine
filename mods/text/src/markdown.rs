//! The subset of Markdown the text operations need: ATX headings and fenced
//! code blocks. Headings inside a fence are ignored.

/// An ATX heading found outside fenced code blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Heading {
    pub level: u8,
    pub text: String,
    /// Byte offset of the heading line in the scanned text.
    pub offset: usize,
}

/// A region of a heading split: the preamble before the first boundary
/// heading, or the text starting at a boundary heading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region<'a> {
    /// The heading that starts the region; `None` for the preamble.
    pub heading: Option<Heading>,
    /// The region text, from its first byte to the next boundary heading (or
    /// the end of the document). The preamble starts at byte 0.
    pub text: &'a str,
}

/// Every ATX heading in `text`, in document order, skipping fenced code.
pub fn headings(text: &str) -> Vec<Heading> {
    let mut found = Vec::new();
    let mut fence: Option<(char, usize)> = None;
    for (offset, line) in line_spans(text) {
        if let Some((marker, length)) = fence {
            if is_closing_fence(line, marker, length) {
                fence = None;
            }
            continue;
        }
        if let Some((marker, length)) = opening_fence(line) {
            fence = Some((marker, length));
            continue;
        }
        if let Some((level, heading)) = atx_heading(line) {
            found.push(Heading {
                level,
                text: heading,
                offset,
            });
        }
    }
    found
}

/// Split `text` into regions at every heading whose level is at most `level`.
/// A heading at or above `level` starts a new region, so a target-level
/// heading splits while a higher-level heading keeps its own content. Headings
/// deeper than `level` stay inside the region that contains them. The
/// preamble — everything before the first boundary heading — is its own region,
/// so text outside target-level sections is preserved rather than dropped.
///
/// The result is empty when the document has no heading at or above `level`.
pub fn regions(text: &str, level: u8) -> Vec<Region<'_>> {
    let all = headings(text);
    let boundaries: Vec<&Heading> = all
        .iter()
        .filter(|heading| heading.level <= level)
        .collect();
    let Some(first) = boundaries.first() else {
        return Vec::new();
    };
    let mut regions = Vec::with_capacity(boundaries.len() + 1);
    if first.offset > 0 {
        regions.push(Region {
            heading: None,
            text: &text[..first.offset],
        });
    }
    for (index, heading) in boundaries.iter().enumerate() {
        let end = boundaries
            .get(index + 1)
            .map(|next| next.offset)
            .unwrap_or(text.len());
        regions.push(Region {
            heading: Some((*heading).clone()),
            text: &text[heading.offset..end],
        });
    }
    regions
}

/// `(byte offset, line without its terminator)` for every line.
fn line_spans(text: &str) -> Vec<(usize, &str)> {
    let mut spans = Vec::new();
    let mut offset = 0;
    for raw in text.split_inclusive('\n') {
        let line = raw.strip_suffix('\n').unwrap_or(raw);
        let line = line.strip_suffix('\r').unwrap_or(line);
        spans.push((offset, line));
        offset += raw.len();
    }
    spans
}

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

/// An opening code fence: up to three spaces, then at least three backticks
/// or tildes.
fn opening_fence(line: &str) -> Option<(char, usize)> {
    let indent = indent_of(line);
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let marker = rest.chars().next()?;
    if marker != '`' && marker != '~' {
        return None;
    }
    let length = rest.chars().take_while(|c| *c == marker).count();
    (length >= 3).then_some((marker, length))
}

fn is_closing_fence(line: &str, marker: char, minimum: usize) -> bool {
    let indent = indent_of(line);
    if indent > 3 {
        return false;
    }
    let rest = &line[indent..];
    let length = rest.chars().take_while(|c| *c == marker).count();
    length >= minimum && rest[length..].trim().is_empty()
}

/// An ATX heading: up to three spaces, one to six `#`, a space or the line
/// end, then the text with an optional closing `#` sequence removed.
fn atx_heading(line: &str) -> Option<(u8, String)> {
    let indent = indent_of(line);
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let level = rest.chars().take_while(|c| *c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let after = &rest[level..];
    if !after.is_empty() && !after.starts_with(' ') && !after.starts_with('\t') {
        return None;
    }
    let mut text = after.trim();
    if text.ends_with('#') {
        let run = text.chars().rev().take_while(|c| *c == '#').count();
        let before = &text[..text.len() - run];
        if before.is_empty() || before.ends_with(' ') || before.ends_with('\t') {
            text = before.trim_end();
        }
    }
    Some((level as u8, text.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atx_headings_capture_level_and_text() {
        let headings = headings("# One\n## Two\n###### Six\n####### seven\n#tag\n");
        assert_eq!(
            headings,
            vec![
                Heading {
                    level: 1,
                    text: "One".to_owned(),
                    offset: 0,
                },
                Heading {
                    level: 2,
                    text: "Two".to_owned(),
                    offset: 6,
                },
                Heading {
                    level: 6,
                    text: "Six".to_owned(),
                    offset: 13,
                },
            ]
        );
    }

    #[test]
    fn closing_hashes_are_stripped_only_after_a_space() {
        let headings = headings("# Title ###\n# C#\n# #\n");
        let texts: Vec<&str> = headings.iter().map(|h| h.text.as_str()).collect();
        assert_eq!(texts, ["Title", "C#", ""]);
    }

    #[test]
    fn headings_can_be_indented_up_to_three_spaces() {
        let headings = headings("   ## indented\n    ## code\n");
        assert_eq!(headings.len(), 1);
        assert_eq!(headings[0].text, "indented");
    }

    #[test]
    fn headings_inside_fences_are_ignored() {
        let text =
            "# Real\n```rust\n# fake\n### also fake\n```\n## After\n~~~~\n# tilde fake\n~~~~\n";
        let levels: Vec<u8> = headings(text).iter().map(|h| h.level).collect();
        assert_eq!(levels, [1, 2]);
    }

    #[test]
    fn a_longer_fence_is_not_closed_by_a_shorter_one() {
        let text = "````\n# fake\n```\n# still fake\n````\n# real\n";
        let headings = headings(text);
        assert_eq!(headings.len(), 1);
        assert_eq!(headings[0].text, "real");
    }

    #[test]
    fn regions_start_at_every_heading_at_or_above_the_target_level() {
        let text = "intro\n\n## A\n\none\n\n### A1\n\nsub\n\n## B\n\ntwo\n";
        let regions = regions(text, 2);
        assert_eq!(regions.len(), 3);
        assert_eq!(regions[0].heading, None);
        assert_eq!(regions[0].text, "intro\n\n");
        let first = regions[1].heading.as_ref().expect("heading");
        assert_eq!(first.text, "A");
        assert_eq!(
            regions[1].text, "## A\n\none\n\n### A1\n\nsub\n\n",
            "a deeper heading stays inside its region"
        );
        assert_eq!(regions[2].heading.as_ref().expect("heading").text, "B");
        assert_eq!(regions[2].text, "## B\n\ntwo\n");
    }

    #[test]
    fn higher_level_headings_start_their_own_region() {
        let text = "# Title\n\nintro\n\n## A\n\none\n\n# Part\n\nunder part\n\n## B\n\ntwo\n";
        let regions = regions(text, 2);
        let titles: Vec<Option<&str>> = regions
            .iter()
            .map(|region| region.heading.as_ref().map(|heading| heading.text.as_str()))
            .collect();
        assert_eq!(titles, [Some("Title"), Some("A"), Some("Part"), Some("B")]);
        assert_eq!(regions[2].text, "# Part\n\nunder part\n\n");
    }

    #[test]
    fn regions_are_empty_without_a_heading_at_or_above_the_target_level() {
        assert!(regions("## A\n\n### B\n", 1).is_empty());
        assert!(regions("plain text", 1).is_empty());
        assert!(regions("   \n\n", 2).is_empty());
    }
}
