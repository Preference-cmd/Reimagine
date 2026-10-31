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

/// A section that starts at a heading of a chosen level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section<'a> {
    pub heading: Heading,
    /// The section text, starting with the heading line and ending before the
    /// next heading of the same or a higher level.
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

/// Split `text` into sections at headings of `level`. A section runs until
/// the next heading of the same or a higher level, so subheadings stay inside
/// their parent section. Text before the first heading is not a section.
pub fn sections(text: &str, level: u8) -> Vec<Section<'_>> {
    let all = headings(text);
    let mut sections = Vec::new();
    for (index, heading) in all.iter().enumerate() {
        if heading.level != level {
            continue;
        }
        let end = all[index + 1..]
            .iter()
            .find(|next| next.level <= level)
            .map(|next| next.offset)
            .unwrap_or(text.len());
        sections.push(Section {
            heading: heading.clone(),
            text: &text[heading.offset..end],
        });
    }
    sections
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
    fn sections_run_to_the_next_heading_of_the_same_or_higher_level() {
        let text = "intro\n\n## A\n\none\n\n### A1\n\nsub\n\n## B\n\ntwo\n";
        let sections = sections(text, 2);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading.text, "A");
        assert_eq!(sections[0].text, "## A\n\none\n\n### A1\n\nsub\n\n");
        assert_eq!(sections[1].heading.text, "B");
        assert_eq!(sections[1].text, "## B\n\ntwo\n");
    }

    #[test]
    fn sections_of_an_absent_level_are_empty() {
        assert!(sections("## A\n\n### B\n", 1).is_empty());
    }
}
