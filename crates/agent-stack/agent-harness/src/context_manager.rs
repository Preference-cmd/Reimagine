//! Conversation context management for the agent loop.
//!
//! [`ContextManager`] owns the per-session conversation history and
//! applies a budget-driven rolling window (CM-V2b, decision table M2):
//! the most recent `tail_turns` complete turns are kept verbatim, the
//! older turns are kept back to the soft budget line, and everything
//! older than that is evicted in whole turn units — never splitting a
//! user↔assistant turn or a tool-call↔tool-result pair. V1 did not
//! summarize dropped messages; V2 keeps the same drop semantics until
//! the summarization module (CM-V2c) lands.
//!
//! Token counting is heuristic, without external dependencies. CJK
//! characters (code point above `0x2E80`) average one token per two
//! characters; all other characters average one token per four
//! characters. A fixed overhead of 4 tokens is added per text block (a
//! message with two text blocks pays 8), plus a fixed per-message
//! budget for file blocks (image 1024, other 256).
//!
//! Estimation is behind the [`TokenEstimator`] seam: the heuristic is
//! the default implementation, and an exact counter (e.g. tiktoken-rs,
//! deferred to V3) can be swapped in without touching callers.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::provider::{ContentBlock, Message};

/// Pre-budget token estimate for a single image file block. This is a
/// planning-time figure only — the provider's reported `Usage` is the
/// source of truth for actual consumption.
const IMAGE_TOKENS_PER_BLOCK: usize = 1024;

/// Pre-budget fallback for non-image file blocks (audio, video, ...).
const OTHER_FILE_TOKENS_PER_BLOCK: usize = 256;

/// Default soft-trigger cushion (CM-V2a, decision table M1): compaction
/// is triggered once the stored history exceeds
/// `max_tokens - reserved_tokens`; the hard limit remains `max_tokens`.
const DEFAULT_RESERVED_TOKENS: usize = 16_000;

/// Default verbatim tail length (CM-V2b, decision table M2): the most
/// recent `tail_turns` complete turns are kept word-for-word.
const DEFAULT_TAIL_TURNS: usize = 2;

/// Default compaction target ratio (CM-V2b, decision table M3): a
/// compaction pass aims to leave the history at this fraction of the
/// window budget. Consumed by the summarization module (CM-V2c).
const DEFAULT_COMPACT_TO_RATIO: f32 = 0.6;

/// Default prune threshold (CM-V2b, decision table M2): once old tool
/// outputs accumulate past this many tokens, the prune stage replaces
/// them with placeholders until `prune_target_tokens` is reached.
const DEFAULT_PRUNE_THRESHOLD_TOKENS: usize = 40_000;

/// Default prune target (CM-V2b, decision table M2).
const DEFAULT_PRUNE_TARGET_TOKENS: usize = 20_000;

/// Default per-record truncation limit (CM-V2b, decision table M2):
/// tool results are truncated to this many estimated tokens when
/// committed.
const DEFAULT_RECORD_TRUNCATE_TOKENS: usize = 2_000;

/// Verbatim tail budget, as a fraction of the available window
/// (`max_tokens - reserved_tokens`), clamped to [`TAIL_BUDGET_MIN`] /
/// [`TAIL_BUDGET_MAX`] (CM-V2b, decision table M2).
const TAIL_BUDGET_RATIO: f32 = 0.25;
const TAIL_BUDGET_MIN: usize = 2_000;
const TAIL_BUDGET_MAX: usize = 8_000;

/// Placeholder text that replaces pruned tool outputs.
const TOOL_OUTPUT_PRUNE_PLACEHOLDER: &str = "[tool output pruned]";

/// Suffix appended to tool results truncated at commit time.
const TOOL_OUTPUT_TRUNCATED_SUFFIX: &str = "\n[output truncated]";

/// Token estimation seam (CM-V2a, decision table M1). Callers may
/// install an exact counter later without changing `ContextManager`
/// internals.
///
/// Contract: `estimate_message` returns the full message cost including
/// any fixed per-message overhead (the default heuristic adds 4 per
/// text block); `estimate_text` returns the raw text cost and is used
/// for plain text outside a message envelope. Custom estimators should
/// mirror these semantics.
pub trait TokenEstimator: Send + Sync {
    /// Estimate the tokens of a full message (content blocks plus
    /// fixed overhead).
    fn estimate_message(&self, message: &Message) -> usize;
    /// Estimate the tokens of plain text (no message envelope
    /// overhead).
    fn estimate_text(&self, text: &str) -> usize;
}

/// Default heuristic estimator: CJK 2 chars/token, other 4 chars/token,
/// +4 per text block, fixed budgets for file blocks.
pub struct HeuristicEstimator;

impl TokenEstimator for HeuristicEstimator {
    fn estimate_message(&self, message: &Message) -> usize {
        estimate_message_tokens(message)
    }

    fn estimate_text(&self, text: &str) -> usize {
        estimate_tokens(text)
    }
}

/// Current on-disk session schema version (CM-V2d, decision table M4).
/// Version 2 adds `schema_version`, the `CompactionRecord` summary
/// shape, and `sticky_count`. Version 1 files (no `schema_version`,
/// string-or-null summary) load through a migration path.
const CURRENT_SCHEMA_VERSION: u32 = 2;

/// Serialized snapshot of a session's context written by
/// [`ContextManager::persist`] and read back by
/// [`ContextManager::load`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PersistedSession {
    session_id: String,
    created_at: String,
    history: Vec<Message>,
    compaction_summary: Option<CompactionRecord>,
    /// Number of sticky front messages (the summary message). Restored
    /// so a resumed session's summary is never window-evicted. Old
    /// files without the field (CM-V2b format) load with 0.
    #[serde(default)]
    sticky_count: usize,
    total_tokens: usize,
    /// On-disk schema version. `default` only serves the fallback path
    /// in [`parse_persisted_session`] (CM-V2c files lack the key); the
    /// value is not consumed after parsing.
    #[serde(default)]
    schema_version: u32,
}

/// V1 on-disk shape (pre CM-V2c): string-or-null summary, no
/// `sticky_count`, no `schema_version`. Migrated into
/// [`PersistedSession`] on load. Only `history` is consumed; the other
/// fields exist to keep the serde shape honest.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct PersistedSessionV1 {
    session_id: String,
    created_at: String,
    history: Vec<Message>,
    compaction_summary: Option<String>,
    total_tokens: usize,
}

/// Persistence failure returned by [`ContextManager::persist`] and
/// [`ContextManager::load`].
#[derive(Debug)]
pub enum ContextError {
    Io(std::io::Error),
    Json(serde_json::Error),
    NotFound(String),
    /// The window changed between [`ContextManager::summarize_request`]
    /// and [`ContextManager::apply_summary`]; the summary cannot be
    /// applied to a different conversation (CM-V2c review fix).
    CompactionStale,
    /// [`ContextManager::apply_summary`] received empty (whitespace-
    /// trimmed) text; an empty sticky summary must not be installed
    /// (AC-25/D-6).
    EmptySummary,
    /// The persisted file was written by a newer build than the
    /// running one.
    UnsupportedVersion(u64),
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "context persistence io error: {err}"),
            Self::Json(err) => write!(f, "context persistence json error: {err}"),
            Self::NotFound(session_id) => {
                write!(f, "no persisted context for session {session_id:?}")
            }
            Self::CompactionStale => write!(
                f,
                "compaction window changed between summarize_request and apply_summary; \
                 fall back to plain eviction"
            ),
            Self::EmptySummary => write!(
                f,
                "summary text is empty; refusing to install an empty sticky summary"
            ),
            Self::UnsupportedVersion(version) => write!(
                f,
                "persisted session schema version {version} is newer than the supported \
                 {CURRENT_SCHEMA_VERSION}"
            ),
        }
    }
}

impl std::error::Error for ContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
            Self::NotFound(_)
            | Self::CompactionStale
            | Self::EmptySummary
            | Self::UnsupportedVersion(_) => None,
        }
    }
}

/// Rolling-window context configuration.
pub struct ContextConfig {
    /// Token hard limit for the stored history.
    pub max_tokens: usize,
    /// Session persistence directory. Stored now for AS-03 session
    /// persistence; not used by the V1 context manager.
    pub session_dir: PathBuf,
    /// Soft-trigger cushion (CM-V2a, decision table M1):
    /// `needs_compaction()` fires once the history exceeds
    /// `max_tokens - reserved_tokens`; `is_over_hard_limit()` fires at
    /// `max_tokens`.
    pub reserved_tokens: usize,
    /// Verbatim tail length (CM-V2b): the most recent `tail_turns`
    /// complete turns are kept word-for-word. Coerced to a minimum of 1
    /// (a value of 0 behaves as 1).
    pub tail_turns: usize,
    /// Compaction target ratio (CM-V2b, consumed by CM-V2c): a
    /// compaction pass aims to leave the history at this fraction of
    /// the window budget.
    pub compact_to_ratio: f32,
    /// Prune threshold (CM-V2b): tool outputs older than the budget
    /// line are replaced with placeholders once their accumulated
    /// tokens exceed this.
    pub prune_threshold_tokens: usize,
    /// Prune target (CM-V2b): the prune stage stops once remaining tool
    /// output tokens are at or below this.
    pub prune_target_tokens: usize,
    /// Per-record truncation limit (CM-V2b): tool results are truncated
    /// to this many estimated tokens when committed.
    pub record_truncate_tokens: usize,
}

impl ContextConfig {
    /// Constructor that defaults every window-policy parameter while
    /// keeping struct-literal call sites readable.
    pub fn new(max_tokens: usize, session_dir: PathBuf) -> Self {
        Self {
            max_tokens,
            session_dir,
            reserved_tokens: DEFAULT_RESERVED_TOKENS,
            tail_turns: DEFAULT_TAIL_TURNS,
            compact_to_ratio: DEFAULT_COMPACT_TO_RATIO,
            prune_threshold_tokens: DEFAULT_PRUNE_THRESHOLD_TOKENS,
            prune_target_tokens: DEFAULT_PRUNE_TARGET_TOKENS,
            record_truncate_tokens: DEFAULT_RECORD_TRUNCATE_TOKENS,
        }
    }
}

/// Decomposition of the per-call token budget
/// `max_tokens = system + summary + windowed_history + input`
/// (CM-V2a, decision table M1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetSnapshot {
    pub system_tokens: usize,
    pub summary_tokens: usize,
    pub history_tokens: usize,
    pub input_tokens: usize,
    pub reserved_tokens: usize,
    pub max_tokens: usize,
}

impl BudgetSnapshot {
    /// Sum of every component except the reserved cushion.
    pub fn windowed_total(&self) -> usize {
        self.system_tokens + self.summary_tokens + self.history_tokens + self.input_tokens
    }

    /// `true` when the windowed total exceeds the soft trigger line
    /// (`max_tokens - reserved_tokens`).
    pub fn is_over_budget(&self) -> bool {
        self.windowed_total() > self.max_tokens.saturating_sub(self.reserved_tokens)
    }

    /// `true` when the windowed total exceeds the hard limit.
    pub fn is_over_hard_limit(&self) -> bool {
        self.windowed_total() > self.max_tokens
    }
}

/// Record of one successful summarization compaction (CM-V2c). Persisted
/// with the session so a resumed session restores its summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionRecord {
    pub text: String,
    pub tokens_before: usize,
    pub tokens_after: usize,
    pub created_at: String,
    /// Which compaction attempt of the current failure streak this
    /// record represents (1 = first).
    pub attempt: u32,
    /// Workspace paths of image files referenced by the evicted
    /// messages (PV-03 decision: references survive compaction).
    pub image_refs: Vec<String>,
}

/// Window plan computed without mutating state (CM-V2b): the eviction
/// range `[sticky_end, keep_start)` is what a compaction pass replaces
/// with a summary; `cuts` are the assistant-halves of oversized turns
/// removed inside the keep region. Both feed the summarization prompt
/// and the replacement range — everything that disappears is
/// summarized when compaction runs, and dropped outright when it
/// falls back to plain eviction.
#[derive(Debug, Clone, PartialEq, Eq)]
struct WindowPlan {
    sticky_end: usize,
    keep_start: usize,
    cuts: Vec<(usize, usize)>,
    /// AC-25: absolute history index and character cap of the newest
    /// turn's user text when that turn is user-only and exceeds the
    /// hard limit — there is no assistant message to cut at, so the
    /// window truncates the user text to its head instead. Applied by
    /// [`ContextManager::apply_window`]; a `None` means no truncation.
    tail_user_truncation: Option<(usize, usize)>,
}

impl WindowPlan {
    /// Absolute ranges of messages that will disappear after the plan
    /// is applied: the eviction range plus any cuts.
    fn disappearing_ranges(&self) -> Vec<(usize, usize)> {
        let mut ranges = Vec::with_capacity(1 + self.cuts.len());
        if self.keep_start > self.sticky_end {
            ranges.push((self.sticky_end, self.keep_start));
        }
        ranges.extend(self.cuts.iter().copied().filter(|(start, end)| end > start));
        ranges.sort_unstable();
        ranges
    }
}

/// Consecutive summarization failures that stop compaction attempts
/// (CM-V2c, decision table M3: 防 thrash).
const MAX_CONSECUTIVE_COMPACTION_FAILURES: u32 = 3;

/// Consecutive *transient* summarization failures that stop compaction
/// attempts for the rest of the session (R2-06, DP-4). A persistently
/// flapping provider must not retry the (expensive) summarizer call on
/// every turn — but a single hiccup must not disable compaction either,
/// so the transient counter is separate from
/// [`MAX_CONSECUTIVE_COMPACTION_FAILURES`] and resets on any success.
const MAX_TRANSIENT_COMPACTION_FAILURES: u32 = 5;

/// Summarization prompt head: the domain-structured six-section schema
/// (CM-V2c, decision table M3). The summarizer is instructed to write
/// in the conversation's language and to treat the anchor (previous
/// summary) as an incremental update target.
const SUMMARY_SCHEMA_PROMPT: &str = "\
Summarize the conversation below so that a fresh context can continue \
the work without the original messages. Write the summary in the same \
language as the conversation. Structure it exactly as:

Goal: ...
Constraints: ...
Progress (Done | Active | Blocked): ...
Key Decisions (decision: rationale): ...
Next Steps: ...
Relevant State (workflow/node/model): ...";

/// Anchor prompt: the previous summary is included as the update
/// target; it is excluded from the evicted conversation (迭代锚定摘要,
/// decision table M3).
const ANCHORED_SUMMARY_PROMPT: &str = "\
Anchored summary (update this incrementally; do not repeat its content \
verbatim in the new summary):

";

/// Image reference prompt: workspace paths that must survive
/// compaction even though the image blocks themselves are dropped
/// (PV-03 decision).
const IMAGE_REFS_PROMPT: &str = "\
Image files referenced by the evicted messages (keep these workspace \
paths in Relevant State; do not inline the images):

";

/// Evicted conversation prompt.
const EVICTED_CONVERSATION_PROMPT: &str = "\
Conversation to summarize:

";

/// Render a message as a compact `role: content` line for the
/// summarizer input. Tool results carry their call id; assistant tool
/// calls list name + arguments.
fn render_message_for_summary(message: &Message) -> String {
    let mut line = format!("{}: {}\n", message.role(), message.content());
    if let Some(id) = message.tool_call_id() {
        line.push_str(&format!("  (result of call {id})\n"));
    }
    for call in message.tool_calls() {
        line.push_str(&format!(
            "  (tool call {}: {}({}) → result below)\n",
            call.id(),
            call.name(),
            call.arguments()
        ));
    }
    line
}

/// Manages conversation history for a single agent session.
pub struct ContextManager {
    config: ContextConfig,
    history: Vec<Message>,
    estimator: Box<dyn TokenEstimator>,
    /// Number of messages at the front of `history` that the window
    /// never evicts (pinned goal/constraints and the sticky summary,
    /// CM-V2b decision table M2). Managed by the summarization module
    /// (CM-V2c); public accessors keep the window honest.
    sticky_count: usize,
    /// Record of the last successful summarization compaction.
    compaction_summary: Option<CompactionRecord>,
    /// Consecutive summarization failures since the last success.
    /// At [`MAX_CONSECUTIVE_COMPACTION_FAILURES`] attempts stop (防
    /// thrash, CM-V2c decision table M3).
    consecutive_compaction_failures: u32,
    /// Consecutive *transient* summarization failures since the last
    /// success (R2-06, DP-4). Unlike
    /// [`ContextManager::consecutive_compaction_failures`], this counts
    /// recoverable provider hiccups; at
    /// [`MAX_TRANSIENT_COMPACTION_FAILURES`] attempts stop for the
    /// session, so a flapping provider cannot retry the summarizer call
    /// on every turn. Resets on any successful compaction.
    transient_compaction_failures: u32,
    /// Window plan captured by the last [`ContextManager::summarize_request`].
    /// [`ContextManager::apply_summary`] verifies the plan is unchanged
    /// before replacing the eviction range, so a summary is never
    /// applied to a different window than the one the model summarized.
    request_plan: Option<WindowPlan>,
}

impl ContextManager {
    pub fn new(config: ContextConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            estimator: Box::new(HeuristicEstimator),
            sticky_count: 0,
            compaction_summary: None,
            consecutive_compaction_failures: 0,
            transient_compaction_failures: 0,
            request_plan: None,
        }
    }

    /// Install a custom estimator (exact tokenizer, calibrated model).
    /// The default [`HeuristicEstimator`] remains when unset.
    pub fn with_estimator(mut self, estimator: Box<dyn TokenEstimator>) -> Self {
        self.estimator = estimator;
        self
    }

    /// Number of sticky front messages the window never evicts.
    pub fn sticky_count(&self) -> usize {
        self.sticky_count
    }

    /// Record of the last successful summarization compaction, if any.
    pub fn compaction_summary(&self) -> Option<&CompactionRecord> {
        self.compaction_summary.as_ref()
    }

    /// Consecutive summarization failures since the last success.
    pub fn consecutive_compaction_failures(&self) -> u32 {
        self.consecutive_compaction_failures
    }

    /// Consecutive transient summarization failures since the last
    /// success (R2-06).
    pub fn transient_compaction_failures(&self) -> u32 {
        self.transient_compaction_failures
    }

    /// `true` while the window has messages pending eviction (the
    /// range a compaction pass would summarize). Compaction is only
    /// meaningful when this holds — with a single turn the window
    /// keeps everything and there is nothing to compress.
    pub fn has_eviction_pending(&self) -> bool {
        !self.window_plan().disappearing_ranges().is_empty()
    }

    /// `true` while compaction attempts are allowed. Flipped `false`
    /// after [`MAX_CONSECUTIVE_COMPACTION_FAILURES`] consecutive
    /// permanent failures or [`MAX_TRANSIENT_COMPACTION_FAILURES`]
    /// consecutive transient failures (防 thrash, R2-06) and stays
    /// `false` for the rest of the session (the counters are not
    /// persisted; a restart resets them) — the caller should surface
    /// the error and keep dropping oldest.
    pub fn should_attempt_compaction(&self) -> bool {
        self.consecutive_compaction_failures < MAX_CONSECUTIVE_COMPACTION_FAILURES
            && self.transient_compaction_failures < MAX_TRANSIENT_COMPACTION_FAILURES
    }

    /// Register a failed summarization attempt. The caller falls back
    /// to plain eviction (drop-oldest) and the turn still completes.
    pub fn record_compaction_failure(&mut self) {
        self.consecutive_compaction_failures =
            self.consecutive_compaction_failures.saturating_add(1);
    }

    /// Mark a summarization failure as transient and recoverable
    /// (AC-24, R2-06). Unlike
    /// [`ContextManager::record_compaction_failure`], this does NOT
    /// advance the permanent streak; instead it advances the separate
    /// transient counter, so a single provider hiccup keeps compaction
    /// enabled while a persistently flapping provider stops attempts
    /// after [`MAX_TRANSIENT_COMPACTION_FAILURES`] (bounded retry). A
    /// successful compaction resets both counters.
    pub fn record_transient_compaction_failure(&mut self) {
        self.transient_compaction_failures = self.transient_compaction_failures.saturating_add(1);
    }

    /// Build the summarizer input for the messages the window is about
    /// to evict (CM-V2c): a single `user` message containing the
    /// domain-structured six-section schema prompt, the anchored
    /// previous summary (when present), the workspace image references
    /// of the evicted messages, and the evicted conversation itself.
    ///
    /// The manager never calls a model itself — the caller (loop,
    /// CM-V2e) runs the returned messages through the provider and
    /// feeds the reply text to [`ContextManager::apply_summary`].
    ///
    /// The window plan at call time is captured; [`ContextManager::apply_summary`]
    /// rejects the result with [`ContextError::CompactionStale`] when
    /// the window changed in between (e.g. a turn was committed while
    /// the model was summarizing).
    pub fn summarize_request(&mut self) -> Vec<Message> {
        let plan = self.window_plan();
        self.request_plan = Some(plan.clone());
        let ranges = plan.disappearing_ranges();

        let mut sections: Vec<String> = Vec::new();
        sections.push(SUMMARY_SCHEMA_PROMPT.to_owned());

        if let Some(anchor) = &self.compaction_summary {
            sections.push(format!(
                "{ANCHORED_SUMMARY_PROMPT}\n{anchor_text}",
                anchor_text = anchor.text
            ));
        }

        let refs = self.collect_image_refs(&ranges);
        if !refs.is_empty() {
            sections.push(format!("{IMAGE_REFS_PROMPT}\n{}", refs.join("\n")));
        }

        let mut evicted = String::new();
        for (start, end) in &ranges {
            for message in &self.history[*start..*end] {
                evicted.push_str(&render_message_for_summary(message));
            }
        }
        if !evicted.is_empty() {
            sections.push(format!("{EVICTED_CONVERSATION_PROMPT}\n{evicted}"));
        }

        vec![Message::user(sections.join("\n\n"))]
    }

    /// Apply a successful summarization reply: the eviction range is
    /// replaced by a sticky summary message (never evicted by the
    /// window), the `CompactionRecord` is stored, and the failure
    /// streak resets. Returns the recorded summary.
    ///
    /// `Err(ContextError::CompactionStale)` when the window changed
    /// since [`ContextManager::summarize_request`] — the caller must
    /// fall back to plain eviction (drop-oldest) and the turn still
    /// completes.
    ///
    /// `Err(ContextError::EmptySummary)` when the text is empty after
    /// trimming whitespace (AC-25/D-6) — an empty sticky summary is
    /// never installed. The loop driver's own empty-reply guard is a
    /// harmless redundancy.
    pub fn apply_summary(
        &mut self,
        text: impl Into<String>,
    ) -> Result<CompactionRecord, ContextError> {
        let text = text.into();
        if text.trim().is_empty() {
            return Err(ContextError::EmptySummary);
        }
        let plan = self.window_plan();
        if let Some(requested) = &self.request_plan
            && *requested != plan
        {
            return Err(ContextError::CompactionStale);
        }
        self.request_plan = None;
        let ranges = plan.disappearing_ranges();
        let tokens_before: usize = ranges
            .iter()
            .map(|(start, end)| {
                self.history[*start..*end]
                    .iter()
                    .map(|m| self.estimator.estimate_message(m))
                    .sum::<usize>()
            })
            .sum();

        let summary_message = Message::user(text.clone());
        let tokens_after = self.estimator.estimate_message(&summary_message);
        let image_refs = self.collect_image_refs(&ranges);

        // Remove the evicted messages from the back.
        for (start, end) in ranges.iter().rev() {
            self.history.drain(*start..*end);
        }
        // Single-generation sticky summary (迭代锚定, decision table
        // M3 — 非多代折叠): a previous summary message is replaced in
        // place, keeping the sticky prefix at exactly one summary
        // message. Pinned goal/constraints blocks (future pin API)
        // live behind it and are untouched.
        if self.sticky_count > 0 {
            if let Some(first) = self.history.first_mut() {
                *first = summary_message;
            } else {
                // Defensive (AC-04): `sticky_count` may outlive its
                // history if a persisted file was hand-edited or
                // partially evicted. Install the summary as a fresh
                // sticky prefix instead of panicking on `history[0]`.
                self.history.push(summary_message);
                self.sticky_count = 1;
            }
        } else {
            self.history.insert(0, summary_message);
            self.sticky_count = 1;
        }

        let record = CompactionRecord {
            text,
            tokens_before,
            tokens_after,
            created_at: created_at(),
            attempt: self.consecutive_compaction_failures + 1,
            image_refs,
        };
        self.compaction_summary = Some(record.clone());
        self.consecutive_compaction_failures = 0;
        self.transient_compaction_failures = 0;
        Ok(record)
    }

    /// Workspace references of image file blocks inside `ranges`:
    /// `Url` sources keep their path/URL. Inline `Data` blocks carry no
    /// workspace path (no filename can be attached today) and their
    /// reference is dropped with the block. Used by the summary so
    /// image references survive compaction (PV-03 decision).
    fn collect_image_refs(&self, ranges: &[(usize, usize)]) -> Vec<String> {
        let mut refs: Vec<String> = Vec::new();
        for (start, end) in ranges {
            for message in &self.history[*start..*end] {
                for block in message.blocks() {
                    if let ContentBlock::File(file) = block
                        && file.media_type().starts_with("image/")
                        && let Some(url) = file.source().url()
                    {
                        refs.push(url.to_owned());
                    }
                }
            }
        }
        refs
    }

    /// Build the message vector for a provider call:
    /// `[system, ...sticky, ...windowed_history, ...input]`.
    ///
    /// The stored history is windowed before the vector is built
    /// (CM-V2b): the most recent `tail_turns` complete turns are kept
    /// verbatim, older turns are kept back to the soft budget line
    /// (`max_tokens - reserved_tokens`), and older messages are evicted
    /// in whole turn units. The sticky prefix (summary / pinned blocks)
    /// is never evicted. The system prompt is re-attached on every call
    /// and is never stored.
    pub fn prepare_messages(&mut self, system: &str, input: &[Message]) -> Vec<Message> {
        self.apply_window();

        let mut messages = Vec::with_capacity(self.history.len() + 1 + input.len());
        messages.push(Message::system(system));
        messages.extend(self.history.iter().cloned());
        messages.extend(input.iter().cloned());
        messages
    }

    /// Append a completed turn (user input + assistant output) to the
    /// conversation history. Tool results are truncated to
    /// `record_truncate_tokens` at commit time (CM-V2b).
    pub fn commit_turn(&mut self, messages: &[Message]) {
        // Per-call sequence for fabricated ids (AC-25): id-less tool
        // results truncated in the same commit get distinct
        // `truncated-{n}` ids.
        let mut fabricated_id_seq = 0usize;
        for message in messages {
            if message.role() == "tool" {
                self.history.push(truncate_tool_result(
                    message,
                    self.config.record_truncate_tokens,
                    &mut fabricated_id_seq,
                ));
            } else {
                self.history.push(message.clone());
            }
        }
    }

    /// Estimated total tokens of the stored history.
    pub fn token_count(&self) -> usize {
        self.history
            .iter()
            .map(|m| self.estimator.estimate_message(m))
            .sum()
    }

    /// Estimated tokens of the history message at `index`. Returns
    /// `None` when `index` is outside the stored history.
    pub fn message_tokens(&self, index: usize) -> Option<usize> {
        self.history
            .get(index)
            .map(|m| self.estimator.estimate_message(m))
    }

    /// Budget decomposition of the next provider call: `system`,
    /// `input` and the reserved cushion are supplied by the caller's
    /// perspective (`system_tokens` / `input_tokens`), while
    /// `history_tokens` and `summary_tokens` come from the stored
    /// state (the summary message is included in the history, so
    /// `summary_tokens` reports the last compaction record's cost).
    pub fn budget_snapshot(&self, system_tokens: usize, input_tokens: usize) -> BudgetSnapshot {
        BudgetSnapshot {
            system_tokens,
            summary_tokens: self
                .compaction_summary
                .as_ref()
                .map(|record| record.tokens_after)
                .unwrap_or(0),
            history_tokens: self.token_count(),
            input_tokens,
            reserved_tokens: self.config.reserved_tokens,
            max_tokens: self.config.max_tokens,
        }
    }

    /// `true` when the stored history exceeds the soft trigger line
    /// (`max_tokens - reserved_tokens`); the turn should compact
    /// before sending.
    pub fn needs_compaction(&self) -> bool {
        self.token_count()
            > self
                .config
                .max_tokens
                .saturating_sub(self.config.reserved_tokens)
    }

    /// `true` when the stored history exceeds `max_tokens`; the turn
    /// must compact (or drop) before sending.
    pub fn is_over_hard_limit(&self) -> bool {
        self.token_count() > self.config.max_tokens
    }

    /// Persist this session's context to
    /// `{session_dir}/{session_id}.json`, creating the session
    /// directory if it does not exist.
    pub fn persist(&self, session_id: &str) -> Result<(), ContextError> {
        let state = PersistedSession {
            session_id: session_id.to_owned(),
            created_at: created_at(),
            history: self.history.clone(),
            compaction_summary: self.compaction_summary.clone(),
            sticky_count: self.sticky_count,
            total_tokens: self.token_count(),
            schema_version: CURRENT_SCHEMA_VERSION,
        };
        std::fs::create_dir_all(&self.config.session_dir).map_err(ContextError::Io)?;
        let path = self.config.session_dir.join(format!("{session_id}.json"));
        let json = serde_json::to_string_pretty(&state).map_err(ContextError::Json)?;
        // Atomic write (AC-23): write to a same-directory temp file,
        // then rename over the target. A reader (or a crash) never
        // observes a partially written session file, and the rename is
        // atomic on the same filesystem.
        let tmp_path = self
            .config
            .session_dir
            .join(format!("{session_id}.json.tmp"));
        std::fs::write(&tmp_path, json).map_err(ContextError::Io)?;
        std::fs::rename(&tmp_path, &path).map_err(|err| {
            // Best-effort cleanup so a failed rename does not leave the
            // temp file behind for the next persist to trip over.
            let _ = std::fs::remove_file(&tmp_path);
            ContextError::Io(err)
        })
    }

    /// Load a session's context previously written by
    /// [`ContextManager::persist`]. The returned manager uses `config`
    /// for the window and token threshold; the history and the
    /// compaction summary are restored (a custom estimator installed
    /// via [`ContextManager::with_estimator`] must be re-applied by the
    /// caller after loading).
    ///
    /// Schema migration (CM-V2d): files without `schema_version` load
    /// as V1 (their string-or-null summary is dropped and no sticky
    /// prefix is restored), falling back to the current shape for
    /// CM-V2c files that carry a record summary. Files written by a
    /// newer build than the running one fail with
    /// [`ContextError::UnsupportedVersion`].
    pub fn load(session_id: &str, config: ContextConfig) -> Result<Self, ContextError> {
        let path = config.session_dir.join(format!("{session_id}.json"));
        let json = std::fs::read(&path).map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                ContextError::NotFound(session_id.to_owned())
            } else {
                ContextError::Io(err)
            }
        })?;
        let (history, compaction_summary, sticky_count) = parse_persisted_session(&json)?;
        // Auto-repair summary<->sticky inconsistency (AC-25/D-8): a
        // file whose summary message is present in the history but was
        // not pinned (`sticky_count: 0`) re-pins it when the first
        // message's content matches the record's summary text exactly —
        // the summary exists, it just lost its pin. A non-matching
        // first message is left untouched (no guessing).
        let sticky_count = if sticky_count == 0
            && let Some(record) = &compaction_summary
            && let Some(first) = history.first()
            && first.content() == record.text
        {
            1
        } else {
            sticky_count
        };
        // Defensive clamp (AC-04): a hand-edited or partially written
        // file may declare more sticky messages than the history holds;
        // `apply_summary` indexes `history[0]` on this count, so it must
        // never exceed the length.
        let sticky_count = sticky_count.min(history.len());
        Ok(Self {
            config,
            history,
            estimator: Box::new(HeuristicEstimator),
            sticky_count,
            compaction_summary,
            consecutive_compaction_failures: 0,
            transient_compaction_failures: 0,
            request_plan: None,
        })
    }

    /// Window the stored history in place (CM-V2b, decision table M2):
    /// prune first, then apply the [`WindowPlan`] computed from the
    /// current state.
    fn apply_window(&mut self) {
        self.prune_old_tool_outputs();
        let plan = self.window_plan();
        // AC-25: truncate an oversized user-only newest turn's text to
        // its head before the range drains, so the plan's absolute
        // indexes stay valid (truncation changes no message count).
        if let Some((idx, max_chars)) = plan.tail_user_truncation
            && let Some(message) = self.history.get_mut(idx)
        {
            *message = truncate_user_message(message, max_chars);
        }
        // Cuts lie inside the keep region: drain from the back so
        // earlier absolute indexes stay valid, then evict the front
        // range.
        for (start, end) in plan.cuts.iter().rev() {
            self.history.drain(*start..*end);
        }
        if plan.keep_start > plan.sticky_end {
            self.history.drain(plan.sticky_end..plan.keep_start);
        }
    }

    /// Compute the window plan without mutating state (CM-V2b):
    ///
    /// 1. **Verbatim tail**: the most recent `tail_turns` complete turns
    ///    are kept word-for-word, bounded by the tail budget
    ///    (`25%` of the available window, 2k-8k clamp).
    /// 2. **Middle window**: turns between the sticky prefix and the
    ///    tail are kept from newest to oldest until the soft budget
    ///    line (`max_tokens - reserved_tokens`) is filled; the rest is
    ///    evicted in whole turn units (a user↔assistant turn and its
    ///    tool-call↔tool-result pairs are never split). A single turn
    ///    larger than the window budget is cut at its first assistant
    ///    message, keeping the user message.
    ///
    /// The plan reports `[sticky_end, keep_start)` as the eviction
    /// range (what summarization replaces) and `cuts` as the
    /// assistant-halves of oversized turns removed inside the keep
    /// region — both feed the summarization prompt and the
    /// replacement range (everything that disappears is summarized
    /// when compaction runs, and dropped when it falls back to plain
    /// eviction).
    fn window_plan(&self) -> WindowPlan {
        let sticky_end = self.sticky_count.min(self.history.len());
        let turns = turn_spans(&self.history[sticky_end..]);

        let window_budget = self
            .config
            .max_tokens
            .saturating_sub(self.config.reserved_tokens);
        // AC-25: the tail budget never exceeds the (positive) soft
        // window — the 2k floor would otherwise refuse to shrink the
        // tail for tiny windows and starve the middle budget to 0. A
        // saturated soft window (`reserved_tokens >= max_tokens`,
        // `window_budget == 0`) has no positive budget to clamp to:
        // the 2k floor stays so the verbatim tail survives, and the
        // hard-limit branch below still caps the newest turn at
        // `max_tokens`.
        let tail_budget = if window_budget > 0 {
            tail_budget(window_budget).min(window_budget)
        } else {
            tail_budget(window_budget)
        };
        let sticky_tokens: usize = (0..sticky_end)
            .map(|i| self.message_tokens(i).unwrap_or(0))
            .sum();

        // Verbatim tail: the most recent `tail_turns` turn spans; drop
        // older tail turns while the tail exceeds its budget (never
        // splits a turn).
        let mut tail_start = turns.len().saturating_sub(self.config.tail_turns.max(1));
        let mut tail_tokens: usize = turns[tail_start..]
            .iter()
            .map(|span| self.span_tokens(sticky_end, span))
            .sum();
        while tail_start + 1 < turns.len() && tail_tokens > tail_budget {
            tail_tokens =
                tail_tokens.saturating_sub(self.span_tokens(sticky_end, &turns[tail_start]));
            tail_start += 1;
        }

        // The newest turn always survives the tail loop. Verbatim-tail
        // protection wins over the soft window, but a turn that alone
        // exceeds the hard limit (`max_tokens`) is cut at its first
        // assistant message so the hard limit still holds; a user-only
        // turn has no assistant message to cut at, so its user text is
        // truncated to the head that fits (AC-25).
        let mut tail_cut: Option<(usize, usize)> = None;
        let mut tail_user_truncation: Option<(usize, usize)> = None;
        if let Some(&newest_span) = turns.get(tail_start)
            && self.span_tokens(sticky_end, &newest_span) > self.config.max_tokens
        {
            let abs_start = sticky_end + newest_span.0;
            // The newest user message keeps its head when it alone is
            // oversized: dropping the newest user instruction entirely
            // would be worse than a truncated head (AC-25). Applied
            // whether or not the turn carries an assistant message — a
            // giant user text plus a short assistant reply used to keep
            // the user text verbatim and blow the hard limit (R2-05).
            let message = &self.history[abs_start];
            // Cost that survives truncation: file block budgets plus
            // the per-text-block overhead (the first text block keeps
            // its overhead even when shortened).
            let fixed_tokens: usize = message
                .blocks()
                .iter()
                .map(|block| match block {
                    ContentBlock::Text(_) => 4,
                    ContentBlock::File(file) if file.media_type().starts_with("image/") => {
                        IMAGE_TOKENS_PER_BLOCK
                    }
                    ContentBlock::File(_) => OTHER_FILE_TOKENS_PER_BLOCK,
                })
                .sum();
            if let Some(ContentBlock::Text(first_text)) = message.blocks().first() {
                // Char cap is conservative for any char mix: ASCII
                // packs 4 chars/token and CJK packs tighter, so an
                // all-ASCII head at the cap is the worst case
                // (estimated tokens == max_tokens, not over).
                let max_chars = self
                    .config
                    .max_tokens
                    .saturating_sub(fixed_tokens)
                    .saturating_mul(4);
                if first_text.chars().count() > max_chars {
                    tail_user_truncation = Some((abs_start, max_chars));
                    let truncated: String = first_text.chars().take(max_chars).collect();
                    tail_tokens = tail_tokens.saturating_sub(
                        self.estimator
                            .estimate_text(first_text)
                            .saturating_sub(self.estimator.estimate_text(&truncated)),
                    );
                }
            }
            // The assistant half is additionally cut when present so the
            // hard limit still holds even with the user head kept
            // (CM-V2e).
            if let Some(off) = first_assistant_offset(&self.history[sticky_end..], newest_span) {
                tail_cut = Some((abs_start + off, abs_start + newest_span.1));
                let removed =
                    self.span_tokens(sticky_end, &(newest_span.0 + off, newest_span.1 - off));
                tail_tokens = tail_tokens.saturating_sub(removed);
            }
        }

        // Middle window: newest-to-oldest turns between the sticky
        // prefix and the tail, filling the remaining budget.
        let mut middle_budget =
            window_budget.saturating_sub(sticky_tokens.saturating_add(tail_tokens));
        let mut keep_from = tail_start;
        while keep_from > 0 {
            let span = turns[keep_from - 1];
            let span_tokens = self.span_tokens(sticky_end, &span);
            if span_tokens > middle_budget {
                break;
            }
            middle_budget -= span_tokens;
            keep_from -= 1;
        }

        // Oversized turn: the first turn that did not fit is cut at its
        // first assistant message when it alone exceeds the window
        // budget, so the user message is still kept. Skipped when the
        // remaining middle budget is exhausted (a user prefix would not
        // fit either — plain eviction is the honest choice).
        let mut oversized_cut: Option<(usize, usize)> = None;
        if keep_from > 0 && middle_budget > 0 {
            let span = turns[keep_from - 1];
            let span_tokens = self.span_tokens(sticky_end, &span);
            if span_tokens > window_budget
                && let Some(off) = first_assistant_offset(&self.history[sticky_end..], span)
            {
                let abs_start = sticky_end + span.0;
                oversized_cut = Some((abs_start + off, abs_start + span.1));
                keep_from -= 1;
            }
        }

        let keep_abs_start = if keep_from >= turns.len() {
            self.history
                .len()
                .min(sticky_end + turns.last().map(|s| s.0 + s.1).unwrap_or(0))
        } else {
            sticky_end + turns[keep_from].0
        };

        let mut cuts: Vec<(usize, usize)> = Vec::new();
        cuts.extend(tail_cut);
        cuts.extend(oversized_cut);

        WindowPlan {
            sticky_end,
            keep_start: keep_abs_start,
            cuts,
            tail_user_truncation,
        }
    }

    /// Estimated tokens of a turn span `(start, len)` whose indexes
    /// are relative to the sticky-offset slice, resolved to absolute
    /// history indexes via `sticky_end`.
    fn span_tokens(&self, sticky_end: usize, span: &(usize, usize)) -> usize {
        (span.0..span.0 + span.1)
            .map(|i| self.message_tokens(sticky_end + i).unwrap_or(0))
            .sum()
    }

    /// Prune stage (CM-V2b): once tool outputs outside the verbatim
    /// tail accumulate past `prune_threshold_tokens`, replace the
    /// oldest of them with placeholders until `prune_target_tokens`
    /// is reached. Replaced outputs keep their `tool_call_id` so
    /// tool-call↔result pairing is preserved.
    fn prune_old_tool_outputs(&mut self) {
        let sticky_end = self.sticky_count.min(self.history.len());
        let turns = turn_spans(&self.history[sticky_end..]);
        let tail_start = turns.len().saturating_sub(self.config.tail_turns.max(1));

        // Only the tool outputs outside the verbatim tail count toward
        // the threshold: the tail is never pruned.
        let tool_tokens: usize = turns[..tail_start]
            .iter()
            .flat_map(|span| span.0..span.0 + span.1)
            .map(|idx| {
                let message = &self.history[sticky_end + idx];
                if message.role() == "tool" {
                    self.estimator.estimate_message(message)
                } else {
                    0
                }
            })
            .sum();
        if tool_tokens <= self.config.prune_threshold_tokens {
            return;
        }

        let placeholder = Message::tool_result(
            crate::provider::ToolCallId::new("pruned"),
            TOOL_OUTPUT_PRUNE_PLACEHOLDER,
        );
        let placeholder_tokens = self.estimator.estimate_message(&placeholder);
        let mut remaining = tool_tokens;
        // Per-call sequence for fabricated ids (AC-25): id-less tool
        // results pruned in the same pass get distinct `pruned-{n}`
        // ids instead of colliding on a single "pruned".
        let mut fabricated_id_seq = 0usize;
        'outer: for span in &turns[..tail_start] {
            for idx in span.0..span.0 + span.1 {
                let absolute = sticky_end + idx;
                if self.history[absolute].role() != "tool" {
                    continue;
                }
                let tokens = self.estimator.estimate_message(&self.history[absolute]);
                // Progress guard (AC-25): a placeholder that costs more
                // than the output it replaces would grow `remaining`
                // and prune the whole non-tail region without ever
                // reaching the target; tiny outputs stay verbatim
                // instead.
                if tokens <= placeholder_tokens {
                    continue;
                }
                let id = match self.history[absolute].tool_call_id() {
                    Some(id) => id.clone(),
                    None => {
                        fabricated_id_seq += 1;
                        crate::provider::ToolCallId::new(format!("pruned-{fabricated_id_seq}"))
                    }
                };
                self.history[absolute] = Message::tool_result(id, TOOL_OUTPUT_PRUNE_PLACEHOLDER);
                remaining = remaining
                    .saturating_sub(tokens)
                    .saturating_add(placeholder_tokens);
                if remaining <= self.config.prune_target_tokens {
                    break 'outer;
                }
            }
        }
    }
}

/// Split `messages` into complete turn spans. A turn starts at a
/// `user` message and runs to the message before the next `user`
/// message; leading non-`user` messages (resume artifacts) join the
/// first turn. Returns `(start_index, len)` pairs.
fn turn_spans(messages: &[Message]) -> Vec<(usize, usize)> {
    if messages.is_empty() {
        return Vec::new();
    }
    let mut spans: Vec<(usize, usize)> = vec![(0, 0)];
    let mut seen_user = messages[0].role() == "user";
    for (idx, message) in messages.iter().enumerate().skip(1) {
        if message.role() == "user" {
            if seen_user {
                let last = spans.last_mut().expect("non-empty");
                last.1 = idx - last.0;
                spans.push((idx, 0));
            } else {
                seen_user = true;
            }
        }
    }
    let last = spans.last_mut().expect("non-empty");
    last.1 = messages.len() - last.0;
    spans
}

/// Offset of the first `assistant` message inside the turn slice, if
/// any — the cut point for a turn that alone exceeds the window
/// budget.
fn first_assistant_offset(messages: &[Message], span: (usize, usize)) -> Option<usize> {
    (0..span.1).find(|offset| messages[span.0 + offset].role() == "assistant")
}

/// Verbatim tail budget: `25%` of the available window, clamped to
/// [`TAIL_BUDGET_MIN`] / [`TAIL_BUDGET_MAX`] (CM-V2b).
fn tail_budget(window_budget: usize) -> usize {
    let ratio = (window_budget as f32 * TAIL_BUDGET_RATIO) as usize;
    ratio.clamp(TAIL_BUDGET_MIN, TAIL_BUDGET_MAX)
}

/// Truncate a `tool` result message to `record_truncate_tokens`
/// estimated tokens at commit time (CM-V2b). The truncation is
/// character-based (an ASCII approximation of the token budget) with
/// the truncated marker appended; messages already within budget are
/// returned unchanged.
///
/// A `tool` result without a `tool_call_id` gets a fabricated
/// `truncated-{n}` id (AC-25); `fabricated_id_seq` is the per-call
/// sequence counter that keeps such ids distinct within one commit.
fn truncate_tool_result(
    message: &Message,
    record_truncate_tokens: usize,
    fabricated_id_seq: &mut usize,
) -> Message {
    if message.role() != "tool" {
        return message.clone();
    }
    let content = message.content();
    let text = match message.blocks().first() {
        Some(ContentBlock::Text(text)) => text.as_str(),
        _ => content,
    };
    let limit_chars = record_truncate_tokens.saturating_mul(4);
    if text.chars().count() <= limit_chars {
        return message.clone();
    }
    let truncated: String = text.chars().take(limit_chars).collect();
    let id = match message.tool_call_id() {
        Some(id) => id.clone(),
        None => {
            *fabricated_id_seq += 1;
            crate::provider::ToolCallId::new(format!("truncated-{fabricated_id_seq}"))
        }
    };
    Message::tool_result(id, format!("{truncated}{TOOL_OUTPUT_TRUNCATED_SUFFIX}"))
}

/// Truncate the leading `Text` block of `message` to `max_chars`
/// characters, keeping the message head and any non-text blocks
/// (AC-25). Used when an oversized user-only newest turn has no
/// assistant message to cut at.
fn truncate_user_message(message: &Message, max_chars: usize) -> Message {
    let mut blocks = message.blocks().to_vec();
    if let Some(ContentBlock::Text(text)) = blocks.first_mut() {
        *text = text.chars().take(max_chars).collect();
    }
    message.clone().with_blocks(blocks)
}

/// Parse a persisted session file with schema migration (CM-V2d):
/// files declaring `schema_version` parse as the current shape; files
/// without it are treated as V1 (string-or-null summary dropped, no
/// sticky prefix) — unless that fails, in which case they are parsed
/// as the current shape (files written by CM-V2c had no
/// `schema_version` yet carried a `CompactionRecord` summary).
/// Returns `(history, compaction_summary, sticky_count)`.
fn parse_persisted_session(
    json: &[u8],
) -> Result<(Vec<Message>, Option<CompactionRecord>, usize), ContextError> {
    let envelope: serde_json::Value = serde_json::from_slice(json).map_err(ContextError::Json)?;
    let version = envelope.get("schema_version").and_then(|v| v.as_u64());
    match version {
        Some(v) if v > CURRENT_SCHEMA_VERSION as u64 => {
            return Err(ContextError::UnsupportedVersion(v));
        }
        Some(_) => {
            let state: PersistedSession =
                serde_json::from_value(envelope).map_err(ContextError::Json)?;
            return Ok((state.history, state.compaction_summary, state.sticky_count));
        }
        None => {}
    }
    // No schema_version: try V1 first; fall back to the current shape
    // for CM-V2c files (record summary, sticky_count present).
    if let Ok(v1) = serde_json::from_value::<PersistedSessionV1>(envelope.clone()) {
        return Ok((v1.history, None, 0));
    }
    let state: PersistedSession = serde_json::from_value(envelope).map_err(ContextError::Json)?;
    Ok((state.history, state.compaction_summary, state.sticky_count))
}

/// Unix epoch seconds used as the `created_at` persisted field.
fn created_at() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs()
        .to_string()
}

/// Estimate the token count of a message from its content blocks: text
/// blocks use the character heuristic below, file blocks use fixed
/// per-block budgets ([`IMAGE_TOKENS_PER_BLOCK`] for images,
/// [`OTHER_FILE_TOKENS_PER_BLOCK`] otherwise).
fn estimate_message_tokens(message: &Message) -> usize {
    message
        .blocks()
        .iter()
        .map(|block| match block {
            ContentBlock::Text(text) => estimate_tokens(text),
            ContentBlock::File(file) if file.media_type().starts_with("image/") => {
                IMAGE_TOKENS_PER_BLOCK
            }
            ContentBlock::File(_) => OTHER_FILE_TOKENS_PER_BLOCK,
        })
        .sum()
}

/// Estimate the token count of `text`.
///
/// Counting is character-based (`chars().count()`), so multi-byte
/// non-CJK characters such as Cyrillic, Arabic, and accented Latin are
/// handled correctly and fall into the "other" bucket.
pub(crate) fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    let cjk = text.chars().filter(|c| *c as u32 > 0x2E80).count();
    let other = chars - cjk;
    (cjk / 2) + (other / 4) + 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::provider::{ContentBlock, FileContentBlock, ToolCall, ToolCallId};

    fn config() -> ContextConfig {
        ContextConfig::new(10_000, PathBuf::from("/tmp/reimagine-test"))
    }

    #[test]
    fn prepare_keeps_full_history_within_window() {
        let mut manager = ContextManager::new(config());
        manager.commit_turn(&[Message::user("u1"), Message::assistant("a1")]);
        manager.commit_turn(&[Message::user("u2"), Message::assistant("a2")]);

        let messages = manager.prepare_messages("sys", &[Message::user("u3")]);
        assert_eq!(messages.len(), 6);
        assert_eq!(messages[0].role(), "system");
        assert_eq!(messages[0].content(), "sys");
        assert_eq!(messages[1].content(), "u1");
        assert_eq!(messages[2].content(), "a1");
        assert_eq!(messages[3].content(), "u2");
        assert_eq!(messages[4].content(), "a2");
        assert_eq!(messages[5].content(), "u3");
    }

    #[test]
    fn prepare_drops_oldest_messages_beyond_window() {
        let mut manager = ContextManager::new(config());
        for i in 0..3 {
            manager.commit_turn(&[
                Message::user(format!("u{i}")),
                Message::assistant(format!("a{i}")),
            ]);
        }

        // Window is 4; the oldest 2 of 6 history messages are dropped.
        let messages = manager.prepare_messages("sys", &[Message::user("u3")]);
        assert_eq!(messages.len(), 6);
        assert_eq!(messages[0].content(), "sys");
        assert_eq!(messages[1].content(), "u1");
        assert_eq!(messages[2].content(), "a1");
        assert_eq!(messages[3].content(), "u2");
        assert_eq!(messages[4].content(), "a2");
        assert_eq!(messages[5].content(), "u3");
    }

    #[test]
    fn prepare_with_empty_input_still_attaches_system() {
        let mut manager = ContextManager::new(config());
        let messages = manager.prepare_messages("sys", &[]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), "system");
        assert_eq!(messages[0].content(), "sys");
    }

    #[test]
    fn commit_turn_accumulates_history() {
        let mut manager = ContextManager::new(config());
        assert_eq!(manager.token_count(), 0);
        manager.commit_turn(&[Message::user("u1"), Message::assistant("a1")]);
        manager.commit_turn(&[Message::user("u2")]);

        // A single character counts the 4-token per-message overhead.
        assert_eq!(manager.token_count(), 12);
    }

    #[test]
    fn token_count_sums_history_content() {
        let mut manager = ContextManager::new(config());
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        assert_eq!(manager.token_count(), 8);
    }

    #[test]
    fn needs_compaction_flips_when_tokens_exceed_soft_trigger() {
        // max_tokens 100, reserved 90 -> soft line at 10.
        let mut manager = ContextManager::new(ContextConfig {
            reserved_tokens: 90,
            ..ContextConfig::new(100, PathBuf::from("/tmp/reimagine-test"))
        });
        assert!(!manager.needs_compaction());
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        // 4 + 4 = 8 <= 10: under the soft line.
        assert!(!manager.needs_compaction());
        manager.commit_turn(&[Message::user("c"), Message::assistant("d")]);
        // 16 > 10: soft trigger fires.
        assert!(manager.needs_compaction());
    }

    #[test]
    fn is_over_hard_limit_flips_at_max_tokens() {
        let mut manager =
            ContextManager::new(ContextConfig::new(20, PathBuf::from("/tmp/reimagine-test")));
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        // 8 <= 20; soft line (20 - 16k -> 0) already fired, hard has not.
        assert!(manager.needs_compaction());
        assert!(!manager.is_over_hard_limit());
        manager.commit_turn(&[Message::user("c"), Message::assistant("d")]);
        // 16 <= 20: still under the hard limit.
        assert!(!manager.is_over_hard_limit());
        manager.commit_turn(&[Message::user("e"), Message::assistant("f")]);
        // 24 > 20: hard limit fires.
        assert!(manager.is_over_hard_limit());
    }

    #[test]
    fn soft_trigger_precedes_hard_limit() {
        let mut manager = ContextManager::new(ContextConfig {
            reserved_tokens: 40,
            ..ContextConfig::new(100, PathBuf::from("/tmp/reimagine-test"))
        });
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        // 8 < 60: neither fires.
        assert!(!manager.needs_compaction());
        assert!(!manager.is_over_hard_limit());
        for i in 0..6 {
            manager.commit_turn(&[
                Message::user(format!("u{i}")),
                Message::assistant(format!("a{i}")),
            ]);
        }
        // 7 turns = 56 tokens: soft (60) not yet fired, hard (100) not.
        assert!(!manager.needs_compaction());
        manager.commit_turn(&[Message::user("u7"), Message::assistant("a7")]);
        // 64 > 60: soft fires; 64 < 100: hard still quiet.
        assert!(manager.needs_compaction());
        assert!(!manager.is_over_hard_limit());
    }

    #[test]
    fn new_config_defaults_reserved_cushion() {
        let config = ContextConfig::new(64_000, PathBuf::from("/tmp/reimagine-test"));
        assert_eq!(config.reserved_tokens, 16_000);
    }

    #[test]
    fn budget_snapshot_decomposes_call_budget() {
        let mut manager = ContextManager::new(ContextConfig {
            reserved_tokens: 40,
            ..ContextConfig::new(100, PathBuf::from("/tmp/reimagine-test"))
        });
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        let snapshot = manager.budget_snapshot(/* system */ 20, /* input */ 12);
        assert_eq!(snapshot.system_tokens, 20);
        assert_eq!(snapshot.summary_tokens, 0);
        assert_eq!(snapshot.history_tokens, 8);
        assert_eq!(snapshot.input_tokens, 12);
        assert_eq!(snapshot.reserved_tokens, 40);
        assert_eq!(snapshot.max_tokens, 100);
        assert_eq!(snapshot.windowed_total(), 40);
        // Soft line = 100 - 40 = 60; hard line = 100.
        assert!(!snapshot.is_over_budget());
        assert!(!snapshot.is_over_hard_limit());

        manager.commit_turn(&[Message::user("u1"), Message::assistant("a1")]);
        let snapshot = manager.budget_snapshot(20, 12);
        assert_eq!(snapshot.windowed_total(), 48);
        assert!(!snapshot.is_over_budget());

        manager.commit_turn(&[Message::user("u2"), Message::assistant("a2")]);
        let snapshot = manager.budget_snapshot(20, 12);
        // system 20 + history 24 (3 turns) + input 12 = 56 <= 60: quiet.
        assert_eq!(snapshot.windowed_total(), 56);
        assert!(!snapshot.is_over_budget());
    }

    #[test]
    fn budget_snapshot_flips_over_budget_with_large_history() {
        let mut manager = ContextManager::new(ContextConfig {
            reserved_tokens: 40,
            ..ContextConfig::new(100, PathBuf::from("/tmp/reimagine-test"))
        });
        for i in 0..10 {
            manager.commit_turn(&[
                Message::user(format!("u{i}")),
                Message::assistant(format!("a{i}")),
            ]);
        }
        // 20 messages * 4 tokens = 80. With system 20 + input 0: 100 > 60 soft.
        let snapshot = manager.budget_snapshot(20, 0);
        assert_eq!(snapshot.history_tokens, 80);
        assert_eq!(snapshot.windowed_total(), 100);
        assert!(snapshot.is_over_budget());
        // 100 > 100 is false: exactly at the hard line.
        assert!(!snapshot.is_over_hard_limit());

        // 20 system + 80 history + 1 input message = 104 > 100 hard.
        let snapshot = manager.budget_snapshot(20, 4);
        assert!(snapshot.is_over_hard_limit());
    }

    #[test]
    fn message_tokens_attributes_per_message() {
        let mut manager = ContextManager::new(ContextConfig::new(
            10_000,
            PathBuf::from("/tmp/reimagine-test"),
        ));
        manager.commit_turn(&[Message::user("abc"), Message::assistant("defg")]);
        // "abc" = 0 (3 chars / 4) + 4 overhead; "defg" = 1 + 4.
        assert_eq!(manager.message_tokens(0), Some(4));
        assert_eq!(manager.message_tokens(1), Some(5));
        assert_eq!(manager.message_tokens(2), None);
    }

    struct FlatEstimator(usize);

    impl TokenEstimator for FlatEstimator {
        fn estimate_message(&self, _message: &Message) -> usize {
            self.0
        }

        fn estimate_text(&self, _text: &str) -> usize {
            self.0
        }
    }

    #[test]
    fn custom_estimator_replaces_heuristic() {
        let mut manager = ContextManager::new(ContextConfig::new(
            10_000,
            PathBuf::from("/tmp/reimagine-test"),
        ))
        .with_estimator(Box::new(FlatEstimator(7)));
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        assert_eq!(manager.token_count(), 14);
        assert_eq!(manager.message_tokens(0), Some(7));
        assert_eq!(manager.budget_snapshot(7, 7).history_tokens, 14);
    }

    fn window_config(
        max_tokens: usize,
        reserved_tokens: usize,
        tail_turns: usize,
    ) -> ContextConfig {
        ContextConfig {
            tail_turns,
            ..ContextConfig::new(max_tokens, PathBuf::from("/tmp/reimagine-test"))
        }
        .with_reserved(reserved_tokens)
    }

    impl ContextConfig {
        fn with_reserved(mut self, reserved_tokens: usize) -> Self {
            self.reserved_tokens = reserved_tokens;
            self
        }
    }

    fn turn(i: usize, chars: usize) -> Vec<Message> {
        vec![
            Message::user(format!("u{i}:{}", "a".repeat(chars))),
            Message::assistant(format!("a{i}:{}", "b".repeat(chars))),
        ]
    }

    fn tool_turn(i: usize, chars: usize) -> Vec<Message> {
        vec![
            Message::user(format!("u{i}:{}", "a".repeat(chars))),
            Message::assistant_with_tool_calls(
                format!("call{i}"),
                vec![ToolCall::new(
                    ToolCallId::new(format!("call-{i}")),
                    "echo",
                    json!({"x": i}),
                )],
            ),
            Message::tool_result(ToolCallId::new(format!("call-{i}")), "x".repeat(chars)),
            Message::assistant(format!("a{i}:{}", "b".repeat(chars))),
        ]
    }

    fn text_of(message: &Message) -> String {
        message.content().to_owned()
    }

    #[test]
    fn window_keeps_verbatim_tail_and_drops_oldest() {
        // Window budget 8k, tail budget 2k (clamped), tail_turns 2.
        // A turn of 2x400 chars is ~208 tokens; 60 turns ~12.5k tokens.
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }

        let messages = manager.prepare_messages("sys", &[]);
        let history: Vec<Message> = messages[1..].to_vec();
        // The 2 newest turns survive verbatim.
        let last = history.last().expect("non-empty");
        assert_eq!(text_of(last), format!("a59:{}", "b".repeat(400)));
        assert!(history.iter().any(|m| text_of(m).starts_with("u59")));
        assert!(history.iter().any(|m| text_of(m).starts_with("u58")));
        // Older turns are evicted; nothing above the window budget.
        assert!(!history.iter().any(|m| text_of(m).starts_with("u0")));
        assert!(!history.iter().any(|m| text_of(m).starts_with("u21")));
        assert!(history.iter().any(|m| text_of(m).starts_with("u22")));
        assert!(manager.token_count() <= 10_000);
    }

    #[test]
    fn window_never_splits_tool_pairs() {
        // Window budget 100 (10k - 9.9k): the tail budget is clamped to
        // the window (AC-25), so a single ~116-token turn cannot fit it
        // verbatim; the tail loop shrinks to the newest turn (tail
        // protection) and the middle cannot fit a turn either. The
        // surviving turn's tool pairs stay intact.
        let mut manager = ContextManager::new(window_config(10_000, 9_900, 2));
        for i in 0..10 {
            manager.commit_turn(&tool_turn(i, 100));
        }

        manager.prepare_messages("sys", &[]);
        let history = manager.history.clone();
        // Only the newest turn survives, tool pairs intact.
        assert_eq!(history.len(), 4);
        assert_eq!(history[0].role(), "user");
        assert_eq!(history[0].content(), format!("u9:{}", "a".repeat(100)));
        assert_eq!(history[1].role(), "assistant");
        assert_eq!(history[2].role(), "tool");
        assert_eq!(history[3].role(), "assistant");
        // No orphaned tool results: every tool message has a matching
        // assistant tool call within the kept window.
        for message in &history {
            if message.role() == "tool" {
                let id = message.tool_call_id().expect("tool id");
                let matched = history.iter().any(|m| {
                    m.role() == "assistant" && m.tool_calls().iter().any(|call| call.id() == id)
                });
                assert!(matched, "orphan tool result {id}");
            }
        }
    }

    #[test]
    fn window_clamps_tail_by_token_budget() {
        // 6 turns of ~1010 tokens each; window budget 2k. The tail
        // (4 turns) exceeds the 2k tail budget and shrinks to the
        // newest turn; the middle cannot fit another turn, so only the
        // newest turn survives.
        let mut manager = ContextManager::new(window_config(10_000, 8_000, 4));
        for i in 0..6 {
            manager.commit_turn(&turn(i, 2_000));
        }

        let messages = manager.prepare_messages("sys", &[]);
        let history: Vec<Message> = messages[1..].to_vec();
        assert!(history.iter().any(|m| text_of(m).starts_with("u5")));
        assert!(!history.iter().any(|m| text_of(m).starts_with("u4")));
        assert!(!history.iter().any(|m| text_of(m).starts_with("u0")));
    }

    #[test]
    fn sticky_prefix_is_never_evicted() {
        // Window budget 500: only sticky + tail fit.
        let mut manager = ContextManager::new(window_config(10_000, 9_500, 2));
        manager.sticky_count = 2;
        manager.commit_turn(&[
            Message::user("pinned-goal"),
            Message::assistant("pinned-plan"),
        ]);
        for i in 0..10 {
            manager.commit_turn(&turn(i, 100));
        }

        let messages = manager.prepare_messages("sys", &[]);
        let history: Vec<Message> = messages[1..].to_vec();
        assert_eq!(history[0].content(), "pinned-goal");
        assert_eq!(history[1].content(), "pinned-plan");
        assert!(history.iter().any(|m| text_of(m).starts_with("u9")));
        assert!(history.iter().any(|m| text_of(m).starts_with("u8")));
        assert!(!history.iter().any(|m| text_of(m).starts_with("u0")));
    }

    #[test]
    fn prune_replaces_old_tool_outputs_keeping_ids() {
        // threshold 200 < 3 old tool outputs (~237 tokens) -> prune.
        let mut manager = ContextManager::new(ContextConfig {
            prune_threshold_tokens: 200,
            prune_target_tokens: 100,
            ..window_config(10_000, 2_000, 2)
        });
        for i in 0..5 {
            manager.commit_turn(&tool_turn(i, 300));
        }

        manager.prepare_messages("sys", &[]);
        let history = manager.history.clone();
        // The last 2 turns keep their raw outputs; older ones are
        // placeholders with their tool_call_id preserved.
        let last_turn: Vec<&Message> = history
            .iter()
            .filter(|m| m.content().starts_with("a4"))
            .collect();
        assert!(!last_turn.is_empty());
        let pruned: Vec<&Message> = history
            .iter()
            .filter(|m| m.role() == "tool" && m.content() == TOOL_OUTPUT_PRUNE_PLACEHOLDER)
            .collect();
        assert!(!pruned.is_empty(), "no tool outputs were pruned");
        assert!(pruned.iter().all(|m| m.tool_call_id().is_some()));
        assert!(
            history
                .iter()
                .filter(|m| m.role() == "tool" && m.content().starts_with("xxxxx"))
                .count()
                >= 2
        );
    }

    #[test]
    fn commit_truncates_oversized_tool_results() {
        let mut manager = ContextManager::new(ContextConfig {
            record_truncate_tokens: 10,
            ..window_config(10_000, 2_000, 2)
        });
        manager.commit_turn(&[Message::tool_result(ToolCallId::new("c1"), "x".repeat(100))]);

        let stored = &manager.history[0];
        assert_eq!(stored.role(), "tool");
        assert!(stored.content().ends_with(TOOL_OUTPUT_TRUNCATED_SUFFIX));
        assert_eq!(stored.tool_call_id().map(|id| id.as_str()), Some("c1"));
        assert!(
            stored.content().chars().count() <= 40 + TOOL_OUTPUT_TRUNCATED_SUFFIX.chars().count()
        );
    }

    #[test]
    fn oversized_turn_is_cut_at_first_assistant_keeping_user() {
        // A single turn larger than the whole window budget (~8k): its
        // assistant half is cut, the user message survives.
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        manager.commit_turn(&[
            Message::user("huge:".to_owned() + &"a".repeat(16_000)),
            Message::assistant("b".repeat(16_000)),
        ]);
        manager.commit_turn(&turn(1, 100));
        manager.commit_turn(&turn(2, 100));

        let messages = manager.prepare_messages("sys", &[]);
        let history: Vec<Message> = messages[1..].to_vec();
        assert!(history.iter().any(|m| m.content().starts_with("huge:")));
        assert!(!history.iter().any(|m| m.content() == "b".repeat(16_000)));
    }

    #[test]
    fn newest_turn_alone_over_budget_is_cut_at_assistant() {
        // Review M1 fix: the newest turn survives the tail loop even
        // when it alone exceeds the window budget; a turn beyond the
        // hard limit is cut at its first assistant message instead of
        // blowing the hard limit.
        let mut manager = ContextManager::new(window_config(8_000, 2_000, 2));
        manager.commit_turn(&turn(0, 100));
        manager.commit_turn(&[
            Message::user("newest:".to_owned() + &"a".repeat(16_000)),
            Message::assistant("b".repeat(16_000)),
        ]);

        let messages = manager.prepare_messages("sys", &[]);
        let history: Vec<Message> = messages[1..].to_vec();
        assert!(history.iter().any(|m| m.content().starts_with("newest:")));
        assert!(!history.iter().any(|m| m.content() == "b".repeat(16_000)));
        assert!(manager.token_count() <= 8_000);
    }

    #[test]
    fn user_only_turn_over_hard_limit_is_truncated_to_fit() {
        // AC-25: a newest turn with no assistant message (user-only)
        // that exceeds the hard limit cannot be cut at an assistant
        // half. The user text is truncated to the head that fits the
        // hard limit instead of blowing it.
        let mut manager = ContextManager::new(window_config(8_000, 2_000, 2));
        manager.commit_turn(&turn(0, 100));
        manager.commit_turn(&[Message::user("huge:".to_owned() + &"a".repeat(60_000))]);

        manager.prepare_messages("sys", &[]);
        let history = manager.history.clone();
        // The newest user message survives, but only its head.
        assert!(history.iter().any(|m| m.content().starts_with("huge:")));
        let newest = history
            .iter()
            .find(|m| m.content().starts_with("huge:"))
            .expect("newest user message kept");
        assert!(
            newest.content().chars().count() < 60_006,
            "oversized user text must be truncated"
        );
        assert!(manager.token_count() <= 8_000);
        // The smaller older turn was evicted to make room.
        assert!(!history.iter().any(|m| m.content().starts_with("u0:")));
    }

    #[test]
    fn oversized_user_text_with_assistant_half_is_truncated() {
        // R2-05: a turn whose user text alone exceeds the hard limit,
        // with an assistant half present, used to keep the giant user
        // text verbatim — cutting only the assistant cannot bring the
        // window under the hard limit. The user text is now truncated
        // to its head as well.
        let mut manager = ContextManager::new(window_config(8_000, 2_000, 2));
        manager.commit_turn(&turn(0, 100));
        manager.commit_turn(&[
            Message::user("huge:".to_owned() + &"a".repeat(60_000)),
            Message::assistant("ok"),
        ]);

        manager.prepare_messages("sys", &[]);
        let history = manager.history.clone();
        let newest = history
            .iter()
            .find(|m| m.content().starts_with("huge:"))
            .expect("newest user message kept");
        assert!(
            newest.content().chars().count() < 60_006,
            "oversized user text must be truncated even with an assistant half"
        );
        assert!(
            !history.iter().any(|m| m.content() == "ok"),
            "assistant half is cut (hard-limit invariant)"
        );
        assert!(manager.token_count() <= 8_000);
    }

    #[test]
    fn prune_with_tiny_outputs_does_not_thrash() {
        // AC-25: when placeholders cost more than the outputs they
        // would replace, pruning cannot reach the target; the loop must
        // stop instead of pruning the whole non-tail region. Tiny
        // outputs stay verbatim.
        let mut manager = ContextManager::new(ContextConfig {
            prune_threshold_tokens: 4,
            prune_target_tokens: 0,
            ..window_config(10_000, 2_000, 2)
        });
        for i in 0..4 {
            manager.commit_turn(&[
                Message::user(format!("u{i}")),
                Message::assistant(format!("a{i}")),
                Message::tool_result(ToolCallId::new(format!("c{i}")), "x"),
                Message::assistant(format!("a{i}b")),
            ]);
        }

        manager.prepare_messages("sys", &[]);
        let history = manager.history.clone();
        // None of the ~4-token outputs was replaced by the ~8-token
        // placeholder: all four raw outputs survive.
        let raw = history
            .iter()
            .filter(|m| m.role() == "tool" && m.content() == "x")
            .count();
        assert_eq!(raw, 4, "tiny outputs must not be replaced");
        assert!(
            !history
                .iter()
                .any(|m| m.role() == "tool" && m.content() == TOOL_OUTPUT_PRUNE_PLACEHOLDER),
            "no placeholders may be installed when pruning would grow the footprint"
        );
    }

    #[test]
    fn prune_fabricates_unique_ids_for_idless_tool_results() {
        // AC-25: id-less tool results pruned in the same pass get
        // distinct fabricated ids (previously every one collided on a
        // single "pruned" id).
        let idless_tool_result = |text: &str| -> Message {
            serde_json::from_str(&format!(
                r#"{{"role":"tool","content":[{{"type":"text","text":"{text}"}}],"tool_call_id":null,"tool_calls":[]}}"#
            ))
            .expect("deserialize id-less tool result")
        };
        let mut manager = ContextManager::new(ContextConfig {
            prune_threshold_tokens: 100,
            prune_target_tokens: 0,
            ..window_config(10_000, 2_000, 2)
        });
        // 4 turns: the newest 2 are the verbatim tail, the oldest 2 are
        // the prune region. Tool results are oversized and carry no id.
        for i in 0..4 {
            manager.commit_turn(&[
                Message::user(format!("u{i}")),
                Message::assistant(format!("a{i}")),
                idless_tool_result(&"x".repeat(200)),
                Message::assistant(format!("a{i}b")),
            ]);
        }

        manager.prepare_messages("sys", &[]);
        let pruned: Vec<&Message> = manager
            .history
            .iter()
            .filter(|m| m.role() == "tool" && m.content() == TOOL_OUTPUT_PRUNE_PLACEHOLDER)
            .collect();
        assert!(pruned.len() >= 2, "expected at least two pruned outputs");
        let ids: Vec<&str> = pruned
            .iter()
            .map(|m| m.tool_call_id().expect("fabricated id").as_str())
            .collect();
        assert!(
            ids.iter().all(|id| id.starts_with("pruned-")),
            "fabricated ids must be namespaced: {ids:?}"
        );
        let mut unique = ids.to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            ids.len(),
            "pruned ids must be unique: {ids:?}"
        );
    }

    #[test]
    fn commit_truncation_fabricates_unique_ids_for_idless_tool_results() {
        // AC-25: id-less tool results truncated at commit time get
        // distinct fabricated ids (previously every one collided on a
        // single "truncated" id).
        let idless_tool_result = |text: &str| -> Message {
            serde_json::from_str(&format!(
                r#"{{"role":"tool","content":[{{"type":"text","text":"{text}"}}],"tool_call_id":null,"tool_calls":[]}}"#
            ))
            .expect("deserialize id-less tool result")
        };
        let mut manager = ContextManager::new(ContextConfig {
            record_truncate_tokens: 10,
            ..window_config(10_000, 2_000, 2)
        });
        manager.commit_turn(&[
            idless_tool_result(&"x".repeat(50)),
            idless_tool_result(&"y".repeat(50)),
        ]);

        let truncated: Vec<&Message> = manager
            .history
            .iter()
            .filter(|m| m.content().ends_with(TOOL_OUTPUT_TRUNCATED_SUFFIX))
            .collect();
        assert_eq!(truncated.len(), 2);
        let ids: Vec<&str> = truncated
            .iter()
            .map(|m| m.tool_call_id().expect("fabricated id").as_str())
            .collect();
        assert!(
            ids.iter().all(|id| id.starts_with("truncated-")),
            "fabricated ids must be namespaced: {ids:?}"
        );
        assert_ne!(
            ids[0], ids[1],
            "fabricated truncated ids must be unique: {ids:?}"
        );
    }

    #[test]
    fn newest_turn_within_hard_limit_survives_verbatim() {
        // A newest turn between the window budget and the hard limit is
        // kept verbatim: tail protection wins over the soft window.
        let mut manager = ContextManager::new(window_config(10_000, 9_900, 2));
        manager.commit_turn(&turn(0, 100));
        manager.commit_turn(&[
            Message::user("big:".to_owned() + &"a".repeat(900)),
            Message::assistant("b".repeat(900)),
        ]);

        let messages = manager.prepare_messages("sys", &[]);
        let history: Vec<Message> = messages[1..].to_vec();
        assert!(history.iter().any(|m| m.content().starts_with("big:")));
        assert!(
            history
                .iter()
                .any(|m| m.content().starts_with("b".repeat(900).as_str()))
        );
    }

    #[test]
    fn prune_ignores_tail_tool_outputs_for_threshold() {
        // The threshold counts only tool outputs outside the tail: a
        // huge tool output in the newest turn does not trigger a prune
        // of the older turns.
        let mut manager = ContextManager::new(ContextConfig {
            prune_threshold_tokens: 100,
            prune_target_tokens: 10,
            ..window_config(10_000, 2_000, 2)
        });
        manager.commit_turn(&[Message::user("u0"), Message::assistant("a0")]);
        // Latest turn carries a giant tool output (inside the tail).
        manager.commit_turn(&[
            Message::user("u1"),
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(ToolCallId::new("c1"), "echo", json!({}))],
            ),
            Message::tool_result(ToolCallId::new("c1"), "x".repeat(2_000)),
            Message::assistant("a1"),
        ]);

        manager.prepare_messages("sys", &[]);
        let history = manager.history.clone();
        // The tail tool output keeps its raw content (not pruned).
        assert!(
            history
                .iter()
                .any(|m| m.role() == "tool" && m.content().chars().count() > 1_000)
        );
    }

    #[test]
    fn turn_spans_group_by_user_messages() {
        let messages = vec![
            Message::user("u1"),
            Message::assistant("a1"),
            Message::user("u2"),
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(ToolCallId::new("c1"), "echo", json!({}))],
            ),
            Message::tool_result(ToolCallId::new("c1"), "r"),
            Message::assistant("a2"),
        ];
        let spans = turn_spans(&messages);
        assert_eq!(spans, vec![(0, 2), (2, 4)]);
    }

    #[test]
    fn turn_spans_merge_leading_non_user_messages() {
        let messages = vec![
            Message::assistant("a0"),
            Message::user("u1"),
            Message::assistant("a1"),
        ];
        let spans = turn_spans(&messages);
        assert_eq!(spans, vec![(0, 3)]);
    }

    #[test]
    fn empty_history_windows_to_empty_output() {
        let mut manager = ContextManager::new(window_config(10_000, 9_500, 2));
        let messages = manager.prepare_messages("sys", &[]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), "system");
    }

    #[test]
    fn empty_string_counts_overhead_only() {
        assert_eq!(estimate_tokens(""), 4);
    }

    #[test]
    fn pure_english_is_four_chars_per_token() {
        let text = "a".repeat(1000);
        assert_eq!(estimate_tokens(&text), 254);
    }

    #[test]
    fn pure_cjk_is_two_chars_per_token() {
        let text = "界".repeat(1000);
        assert_eq!(estimate_tokens(&text), 504);
    }

    #[test]
    fn mixed_combines_both_estimators() {
        let mut text = "a".repeat(400);
        text.push_str(&"界".repeat(600));
        assert_eq!(estimate_tokens(&text), 404);
    }

    #[test]
    fn whitespace_heavy_counts_as_other() {
        let text = " ".repeat(1000);
        assert_eq!(estimate_tokens(&text), 254);
    }

    #[test]
    fn cyrillic_counts_as_other() {
        assert_eq!(estimate_tokens("Привет мир"), 6);
    }

    #[test]
    fn image_block_counts_fixed_budget() {
        let mut manager = ContextManager::new(config());
        manager.commit_turn(&[Message::user_with_blocks(vec![
            ContentBlock::Text("look".into()),
            ContentBlock::File(FileContentBlock::data("image/png", "AAAA")),
        ])]);
        // "look": 4 chars -> 1 + 4 overhead; plus the image budget.
        assert_eq!(manager.token_count(), 5 + IMAGE_TOKENS_PER_BLOCK);
    }

    #[test]
    fn non_image_file_block_counts_fallback_budget() {
        let mut manager = ContextManager::new(config());
        manager.commit_turn(&[Message::user_with_blocks(vec![ContentBlock::File(
            FileContentBlock::data("audio/mpeg", "AAAA"),
        )])]);
        // No text block -> no character estimate; only the fallback.
        assert_eq!(manager.token_count(), OTHER_FILE_TOKENS_PER_BLOCK);
    }

    #[test]
    fn text_only_history_is_unchanged_by_blocks_path() {
        let mut manager = ContextManager::new(config());
        manager.commit_turn(&[Message::user("a"), Message::assistant("b")]);
        assert_eq!(manager.token_count(), 8);
    }

    #[test]
    fn arabic_counts_as_other() {
        assert_eq!(estimate_tokens("مرحبا بالعالم"), 7);
    }

    fn temp_session_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-agent-context-{prefix}-{nonce}"))
    }

    #[test]
    fn persist_load_round_trips_history() {
        let dir = temp_session_dir("round-trip");
        let cfg = ContextConfig::new(10_000, dir.clone());
        let mut manager = ContextManager::new(cfg);
        manager.commit_turn(&[
            Message::user("u1"),
            Message::assistant("a1"),
            Message::user("u2"),
            Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    ToolCallId::new("c1"),
                    "echo",
                    json!({"x": 1}),
                )],
            ),
        ]);
        manager.persist("sess-1").expect("persist failed");

        let loaded =
            ContextManager::load("sess-1", ContextConfig::new(10_000, dir)).expect("load failed");
        assert_eq!(loaded.token_count(), manager.token_count());
        assert_eq!(loaded.history, manager.history);
    }

    #[test]
    fn persist_round_trips_file_blocks() {
        let dir = temp_session_dir("file-blocks");
        let cfg = ContextConfig::new(10_000, dir.clone());
        let mut manager = ContextManager::new(cfg);
        manager.commit_turn(&[Message::user_with_blocks(vec![
            ContentBlock::Text("describe".into()),
            ContentBlock::File(FileContentBlock::data("image/png", "AAAA")),
        ])]);
        manager.persist("sess-1").expect("persist failed");

        let loaded =
            ContextManager::load("sess-1", ContextConfig::new(10_000, dir)).expect("load failed");
        assert_eq!(loaded.history, manager.history);
        assert_eq!(loaded.token_count(), manager.token_count());
    }

    #[test]
    fn load_old_string_content_format_fails() {
        // Development-period format change: files written with the V1
        // `content` string fail to load and must be skipped by callers
        // (the daemon's resume path already logs and skips).
        let dir = temp_session_dir("old-format");
        std::fs::create_dir_all(&dir).expect("mkdir failed");
        std::fs::write(
            dir.join("sess-1.json"),
            r#"{"session_id":"sess-1","created_at":"0","history":[{"role":"user","content":"hi"}],"compaction_summary":null,"total_tokens":4}"#,
        )
        .expect("write failed");
        let result = ContextManager::load("sess-1", ContextConfig::new(10_000, dir));
        assert!(result.is_err());
    }

    #[test]
    fn load_v1_schema_migrates_without_summary() {
        // CM-V2d: a V1 file (no schema_version, null summary) loads
        // with the history restored and no compaction state.
        let dir = temp_session_dir("v1-null");
        std::fs::create_dir_all(&dir).expect("mkdir failed");
        std::fs::write(
            dir.join("sess-1.json"),
            r#"{"session_id":"sess-1","created_at":"0","history":[{"role":"user","content":[{"type":"text","text":"hi"}],"tool_call_id":null,"tool_calls":[]},{"role":"assistant","content":[{"type":"text","text":"yo"}],"tool_call_id":null,"tool_calls":[]}],"compaction_summary":null,"total_tokens":8}"#,        )
        .expect("write failed");
        let loaded = ContextManager::load("sess-1", ContextConfig::new(10_000, dir))
            .expect("v1 load failed");
        assert_eq!(loaded.token_count(), 8);
        assert!(loaded.compaction_summary().is_none());
        assert_eq!(loaded.sticky_count(), 0);
    }

    #[test]
    fn load_v1_schema_with_string_summary_drops_it() {
        // A V1 file with a string summary loads; the summary has no
        // structured shape yet and is dropped.
        let dir = temp_session_dir("v1-string");
        std::fs::create_dir_all(&dir).expect("mkdir failed");
        std::fs::write(
            dir.join("sess-1.json"),
            r#"{"session_id":"sess-1","created_at":"0","history":[{"role":"user","content":[{"type":"text","text":"hi"}],"tool_call_id":null,"tool_calls":[]}],"compaction_summary":"old summary","total_tokens":4}"#,
        )
        .expect("write failed");
        let loaded = ContextManager::load("sess-1", ContextConfig::new(10_000, dir))
            .expect("v1 load failed");
        assert!(loaded.compaction_summary().is_none());
        assert_eq!(loaded.history.len(), 1);
    }

    #[test]
    fn load_future_schema_version_errors() {
        let dir = temp_session_dir("future");
        std::fs::create_dir_all(&dir).expect("mkdir failed");
        std::fs::write(
            dir.join("sess-1.json"),
            r#"{"session_id":"sess-1","created_at":"0","history":[],"compaction_summary":null,"sticky_count":0,"total_tokens":0,"schema_version":999}"#,
        )
        .expect("write failed");
        let err = match ContextManager::load("sess-1", ContextConfig::new(10_000, dir)) {
            Ok(_) => panic!("future version must fail"),
            Err(err) => err,
        };
        assert!(matches!(err, ContextError::UnsupportedVersion(999)));
    }

    #[test]
    fn persist_writes_current_schema_version() {
        let dir = temp_session_dir("schema-version");
        let manager = ContextManager::new(ContextConfig::new(10_000, dir.clone()));
        manager.persist("sess-1").expect("persist failed");

        let json = std::fs::read_to_string(dir.join("sess-1.json")).expect("read failed");
        let state: serde_json::Value = serde_json::from_str(&json).expect("json parse failed");
        assert_eq!(state["schema_version"], 2);
    }

    #[test]
    fn v1_load_then_persist_upgrades_to_v2() {
        // A migrated V1 session re-persists with the current schema
        // version.
        let dir = temp_session_dir("v1-upgrade");
        std::fs::create_dir_all(&dir).expect("mkdir failed");
        std::fs::write(
            dir.join("sess-1.json"),
            r#"{"session_id":"sess-1","created_at":"0","history":[{"role":"user","content":[{"type":"text","text":"hi"}],"tool_call_id":null,"tool_calls":[]}],"compaction_summary":null,"total_tokens":4}"#,
        )
        .expect("write failed");
        let loaded = ContextManager::load("sess-1", ContextConfig::new(10_000, dir.clone()))
            .expect("v1 load failed");
        loaded.persist("sess-1").expect("persist failed");

        let json = std::fs::read_to_string(dir.join("sess-1.json")).expect("read failed");
        let state: serde_json::Value = serde_json::from_str(&json).expect("json parse failed");
        assert_eq!(state["schema_version"], 2);
        assert_eq!(state["history"][0]["content"][0]["text"], "hi");
    }

    #[test]
    fn load_cmv2c_format_without_schema_version_falls_back() {
        // Files written by CM-V2c carry a CompactionRecord summary and
        // sticky_count but no schema_version: they must load via the
        // current-shape fallback, not the V1 path.
        let dir = temp_session_dir("v2c-fallback");
        std::fs::create_dir_all(&dir).expect("mkdir failed");
        std::fs::write(
            dir.join("sess-1.json"),
            r#"{"session_id":"sess-1","created_at":"0","history":[{"role":"user","content":[{"type":"text","text":"hi"}],"tool_call_id":null,"tool_calls":[]}],"compaction_summary":{"text":"gen-1","tokens_before":100,"tokens_after":7,"created_at":"0","attempt":1,"image_refs":["assets/a.png"]},"sticky_count":1,"total_tokens":7}"#,
        )
        .expect("write failed");
        let loaded = ContextManager::load("sess-1", ContextConfig::new(10_000, dir))
            .expect("v2c fallback load failed");
        let record = loaded.compaction_summary().expect("summary lost");
        assert_eq!(record.text, "gen-1");
        assert_eq!(record.image_refs, vec!["assets/a.png"]);
        assert_eq!(loaded.sticky_count(), 1);
    }

    #[test]
    fn future_schema_version_error_message_is_helpful() {
        let dir = temp_session_dir("future-msg");
        std::fs::create_dir_all(&dir).expect("mkdir failed");
        std::fs::write(
            dir.join("sess-1.json"),
            r#"{"session_id":"sess-1","created_at":"0","history":[],"compaction_summary":null,"sticky_count":0,"total_tokens":0,"schema_version":999}"#,
        )
        .expect("write failed");
        let err = match ContextManager::load("sess-1", ContextConfig::new(10_000, dir)) {
            Ok(_) => panic!("future version must fail"),
            Err(err) => err,
        };
        let message = err.to_string();
        assert!(message.contains("999"), "{message}");
        assert!(message.contains("newer"), "{message}");
    }

    #[test]
    fn huge_future_schema_version_is_rejected_without_truncation() {
        // 2^32 + 2 must not be truncated into a valid version.
        let dir = temp_session_dir("future-huge");
        std::fs::create_dir_all(&dir).expect("mkdir failed");
        std::fs::write(
            dir.join("sess-1.json"),
            r#"{"session_id":"sess-1","created_at":"0","history":[],"compaction_summary":null,"sticky_count":0,"total_tokens":0,"schema_version":4294967298}"#,
        )
        .expect("write failed");
        let err = match ContextManager::load("sess-1", ContextConfig::new(10_000, dir)) {
            Ok(_) => panic!("huge version must fail"),
            Err(err) => err,
        };
        assert!(matches!(err, ContextError::UnsupportedVersion(_)), "{err}");
    }

    #[test]
    fn persist_creates_session_dir() {
        let dir = temp_session_dir("mkdir");
        let manager = ContextManager::new(ContextConfig::new(10_000, dir.clone()));
        manager.persist("sess-1").expect("persist failed");
        assert!(dir.join("sess-1.json").is_file());
    }

    #[test]
    fn persist_writes_atomically_and_cleans_up_temp_file() {
        // AC-23: persist writes a same-directory temp file and renames
        // it over the target; no `.tmp` file is left behind and the
        // target parses as a loadable session.
        let dir = temp_session_dir("atomic");
        let mut manager = ContextManager::new(ContextConfig::new(10_000, dir.clone()));
        manager.commit_turn(&[Message::user("u1"), Message::assistant("a1")]);
        manager.persist("sess-1").expect("persist failed");

        let target = dir.join("sess-1.json");
        let tmp = dir.join("sess-1.json.tmp");
        assert!(target.is_file(), "target session file must exist");
        assert!(!tmp.exists(), "temp file must not be left behind");

        // A second persist replaces the existing target atomically.
        manager.commit_turn(&[Message::user("u2")]);
        manager.persist("sess-1").expect("second persist failed");
        assert!(
            !tmp.exists(),
            "temp file must not be left behind after overwrite"
        );

        let loaded = ContextManager::load("sess-1", ContextConfig::new(10_000, dir))
            .expect("target parses as a session");
        assert_eq!(loaded.history.len(), 3);
        assert_eq!(loaded.history[0].content(), "u1");
    }

    #[test]
    fn load_missing_session_returns_err() {
        let dir = temp_session_dir("missing");
        let result = ContextManager::load("nope", ContextConfig::new(10_000, dir));
        assert!(result.is_err());
    }

    #[test]
    fn persist_writes_compaction_summary_none() {
        let dir = temp_session_dir("summary");
        let manager = ContextManager::new(ContextConfig::new(10_000, dir.clone()));
        manager.persist("sess-1").expect("persist failed");

        let json = std::fs::read_to_string(dir.join("sess-1.json")).expect("read failed");
        let state: serde_json::Value = serde_json::from_str(&json).expect("json parse failed");
        assert_eq!(state["session_id"], "sess-1");
        assert_eq!(state["history"], serde_json::Value::Array(vec![]));
        assert!(state["compaction_summary"].is_null());
        assert_eq!(state["total_tokens"], 0);
        assert_eq!(state["sticky_count"], 0);
    }

    #[test]
    fn summarize_request_builds_schema_prompt_from_eviction_range() {
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }

        let request = manager.summarize_request();
        assert_eq!(request.len(), 1);
        let prompt = request[0].content().to_owned();
        // Six-section schema with language instruction.
        for section in [
            "Goal:",
            "Constraints:",
            "Progress (Done | Active | Blocked):",
            "Key Decisions",
            "Next Steps:",
            "Relevant State",
            "same language as the conversation",
        ] {
            assert!(prompt.contains(section), "missing {section:?}");
        }
        // Evicted oldest turn is inside the prompt.
        assert!(prompt.contains("u0:"));
        // The verbatim tail is not part of the prompt.
        assert!(!prompt.contains("u59:"));
    }

    #[test]
    fn summarize_request_anchors_previous_summary() {
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }
        manager
            .apply_summary("summary-one: goal preserved")
            .expect("apply failed");

        let request = manager.summarize_request();
        let prompt = request[0].content().to_owned();
        assert!(prompt.contains("Anchored summary"));
        assert!(prompt.contains("summary-one: goal preserved"));
        // The anchored summary text itself is not repeated as evicted
        // conversation (旧摘要从输入排除).
        assert_eq!(prompt.matches("summary-one: goal preserved").count(), 1);
    }

    #[test]
    fn apply_summary_replaces_evicted_range_with_sticky_message() {
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }

        let record = manager
            .apply_summary("goal: continue artwork refinement")
            .expect("apply failed");
        assert_eq!(record.attempt, 1);
        // tokens_before counts the evicted range only, not all history.
        assert!(record.tokens_before > 0);
        assert!(record.tokens_before < manager.token_count() + record.tokens_after);
        assert!(record.tokens_after > 0);
        assert!(manager.compaction_summary().is_some());
        assert_eq!(manager.sticky_count(), 1);
        assert_eq!(
            manager.history[0].content(),
            "goal: continue artwork refinement"
        );

        // The sticky summary survives the window: repeated prepares do
        // not evict it.
        manager.prepare_messages("sys", &[]);
        assert_eq!(
            manager.history[0].content(),
            "goal: continue artwork refinement"
        );
        assert!(manager.token_count() <= 10_000);
    }

    #[test]
    fn apply_summary_then_window_keeps_anchor_and_tail() {
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }
        manager
            .apply_summary("anchor-summary")
            .expect("apply failed");

        manager.prepare_messages("sys", &[]);
        let history: Vec<Message> = manager.history.clone();
        assert_eq!(history[0].content(), "anchor-summary");
        assert!(history.iter().any(|m| m.content().starts_with("u59")));
        // The oldest evicted turn is gone for good.
        assert!(!history.iter().any(|m| m.content().starts_with("u0")));
    }

    #[test]
    fn apply_summary_with_sticky_count_but_empty_history_does_not_panic() {
        // Defensive guard (AC-04): a hand-edited or partially written
        // persisted file may carry `sticky_count > 0` with no history.
        // `apply_summary` must not index `history[0]` blindly.
        let mut manager = ContextManager::new(config());
        manager.sticky_count = 1; // corrupted / hand-edited persisted state

        let request = manager.summarize_request();
        assert!(!request.is_empty());
        let record = manager
            .apply_summary("summarized after corruption")
            .expect("apply succeeds");
        assert_eq!(manager.history.len(), 1);
        assert_eq!(manager.history[0].content(), "summarized after corruption");
        assert_eq!(manager.sticky_count, 1);
        assert_eq!(record.text, "summarized after corruption");
    }

    #[test]
    fn load_clamps_sticky_count_to_history_length() {
        // A file declaring more sticky messages than the history holds
        // must load with a clamped count instead of panicking later
        // (AC-04).
        let dir = temp_session_dir("sticky-clamp");
        std::fs::create_dir_all(&dir).expect("mkdir failed");
        std::fs::write(
            dir.join("sess-1.json"),
            r#"{"session_id":"sess-1","created_at":"0","history":[
                {"role":"user","content":[{"type":"text","text":"hi"}],"tool_call_id":null,"tool_calls":[]},
                {"role":"assistant","content":[{"type":"text","text":"hello"}],"tool_call_id":null,"tool_calls":[]}
            ],"compaction_summary":null,"sticky_count":5,"total_tokens":8,"schema_version":2}"#,
        )
        .expect("write failed");
        let mut loaded = ContextManager::load("sess-1", ContextConfig::new(10_000, dir.clone()))
            .expect("load failed");
        assert_eq!(loaded.sticky_count, 2);
        // A follow-up compaction on the loaded manager never panics.
        let record = loaded
            .apply_summary("compacted after clamp")
            .expect("apply succeeds");
        assert_eq!(record.text, "compacted after clamp");
    }

    #[test]
    fn load_with_sticky_count_but_empty_history_clamps_to_zero() {
        let dir = temp_session_dir("sticky-empty");
        std::fs::create_dir_all(&dir).expect("mkdir failed");
        std::fs::write(
            dir.join("sess-1.json"),
            r#"{"session_id":"sess-1","created_at":"0","history":[],"compaction_summary":null,"sticky_count":1,"total_tokens":0,"schema_version":2}"#,
        )
        .expect("write failed");
        let mut loaded = ContextManager::load("sess-1", ContextConfig::new(10_000, dir.clone()))
            .expect("load failed");
        assert_eq!(loaded.sticky_count, 0);
        // The next compaction installs a fresh sticky prefix.
        let record = loaded
            .apply_summary("fresh sticky")
            .expect("apply succeeds");
        assert_eq!(loaded.history.len(), 1);
        assert_eq!(record.text, "fresh sticky");
    }

    #[test]
    fn summary_collects_image_refs_from_evicted_messages() {
        let mut manager = ContextManager::new(window_config(10_000, 9_500, 2));
        for i in 0..10 {
            manager.commit_turn(&turn(i, 100));
        }
        // An old evicted turn carries an image reference.
        manager.history[1] = Message::user_with_blocks(vec![
            ContentBlock::Text("img".into()),
            ContentBlock::File(FileContentBlock::url("image/png", "assets/reference.png")),
        ]);

        let request = manager.summarize_request();
        assert!(request[0].content().contains("assets/reference.png"));

        let record = manager
            .apply_summary("summary with image ref")
            .expect("apply failed");
        assert_eq!(record.image_refs, vec!["assets/reference.png"]);
    }

    #[test]
    fn compaction_thrash_stops_after_max_failures() {
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }
        assert!(manager.should_attempt_compaction());
        manager.record_compaction_failure();
        manager.record_compaction_failure();
        assert!(manager.should_attempt_compaction());
        manager.record_compaction_failure();
        assert!(!manager.should_attempt_compaction());
        assert_eq!(manager.consecutive_compaction_failures(), 3);

        // A success resets the streak.
        let record = manager.apply_summary("recovered").expect("apply failed");
        assert_eq!(record.attempt, 4);
        assert_eq!(manager.consecutive_compaction_failures(), 0);
        assert!(manager.should_attempt_compaction());
    }

    #[test]
    fn transient_failures_do_not_disable_compaction() {
        // AC-24 + R2-06 (DP-4): transient (recoverable) summarization
        // failures must not advance the permanent streak that disables
        // compaction at MAX_CONSECUTIVE_COMPACTION_FAILURES — only
        // `record_compaction_failure` counts there. But a persistently
        // flapping provider must not retry the summarizer every turn
        // either: after MAX_TRANSIENT_COMPACTION_FAILURES consecutive
        // transients, attempts stop (bounded retry).
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }
        for _ in 0..4 {
            manager.record_transient_compaction_failure();
        }
        assert!(manager.should_attempt_compaction());
        assert_eq!(manager.consecutive_compaction_failures(), 0);
        manager.record_transient_compaction_failure();
        assert!(
            !manager.should_attempt_compaction(),
            "bounded transient retry stops after the cap (R2-06)"
        );
        assert_eq!(manager.transient_compaction_failures(), 5);

        // Permanent failures still disable compaction independently.
        for _ in 0..3 {
            manager.record_compaction_failure();
        }
        assert_eq!(manager.consecutive_compaction_failures(), 3);
        assert!(!manager.should_attempt_compaction());

        // A success resets both counters and re-enables compaction.
        let record = manager.apply_summary("recovered").expect("apply failed");
        assert_eq!(manager.consecutive_compaction_failures(), 0);
        assert_eq!(manager.transient_compaction_failures(), 0);
        assert!(manager.should_attempt_compaction());
        assert_eq!(record.attempt, 4);
    }

    #[test]
    fn compaction_record_round_trips_persistence() {
        let dir = temp_session_dir("record");
        let cfg = ContextConfig {
            ..ContextConfig::new(10_000, dir.clone())
        };
        let mut manager = ContextManager::new(cfg);
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }
        manager
            .apply_summary("persisted summary")
            .expect("apply failed");
        manager.persist("sess-1").expect("persist failed");

        let loaded =
            ContextManager::load("sess-1", ContextConfig::new(10_000, dir)).expect("load failed");
        let record = loaded.compaction_summary().expect("summary lost");
        assert_eq!(record.text, "persisted summary");
        assert_eq!(loaded.sticky_count(), 1);
        assert_eq!(loaded.history[0].content(), "persisted summary");
        assert_eq!(loaded.consecutive_compaction_failures(), 0);
    }

    #[test]
    fn second_compaction_replaces_single_generation_summary() {
        // 非多代折叠 (decision table M3): repeated compactions keep
        // exactly one sticky summary message, replaced in place.
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }
        manager.apply_summary("gen-1").expect("apply failed");
        assert_eq!(manager.sticky_count(), 1);

        // Fill the window again so a second compaction fires.
        for i in 60..120 {
            manager.commit_turn(&turn(i, 400));
        }
        manager.apply_summary("gen-2").expect("apply failed");

        assert_eq!(manager.sticky_count(), 1);
        assert_eq!(manager.history[0].content(), "gen-2");
        assert!(!manager.history.iter().any(|m| m.content() == "gen-1"));
    }

    #[test]
    fn apply_summary_rejects_stale_window() {
        // Tiny window (100 tokens, AC-25 tail clamp): only a 1-turn
        // tail fits. The interrupt turn is oversized enough to shrink
        // the tail and shift the keep region, so the plan captured by
        // summarize_request goes stale.
        let mut manager = ContextManager::new(window_config(10_000, 9_900, 2));
        for i in 0..10 {
            manager.commit_turn(&turn(i, 100));
        }
        manager.summarize_request();
        // The window changes while the model "summarizes": an oversized
        // new turn is committed, shifting the eviction range.
        manager.commit_turn(&[
            Message::user(format!("interrupt:{}", "a".repeat(160))),
            Message::assistant("ok"),
        ]);

        let err = manager
            .apply_summary("stale summary")
            .expect_err("must reject a stale window");
        assert!(matches!(err, ContextError::CompactionStale));
        // Nothing was replaced.
        assert!(manager.compaction_summary().is_none());
        assert_eq!(manager.sticky_count(), 0);
        assert!(
            manager
                .history
                .iter()
                .any(|m| m.content().starts_with("interrupt:"))
        );
    }

    // ----- apply_summary invariants (AC-16) -----
    //
    // These tests pin the CURRENT behavior of `apply_summary` as the
    // API-level contract. Several behaviors are known gaps guarded in
    // the loop driver (`maybe_compact`) rather than here; each test
    // documents which behavior it pins.

    #[test]
    fn apply_summary_rejects_empty_text() {
        // AC-25/D-6: the API itself rejects empty (whitespace-trimmed)
        // summary text — an empty sticky summary must never be
        // installed. The loop driver's empty-reply guard is a harmless
        // redundancy.
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }
        assert!(manager.has_eviction_pending());

        let err = manager
            .apply_summary("")
            .expect_err("empty text must be rejected");
        assert!(matches!(err, ContextError::EmptySummary));
        let err = manager
            .apply_summary("   \n\t ")
            .expect_err("whitespace-only text must be rejected");
        assert!(matches!(err, ContextError::EmptySummary));
        // Nothing was applied: no sticky summary, no record, and the
        // eviction range is still pending.
        assert!(manager.compaction_summary().is_none());
        assert_eq!(manager.sticky_count(), 0);
        assert!(manager.has_eviction_pending());
    }

    #[test]
    fn apply_summary_without_eviction_range_still_inserts_sticky() {
        // A manager whose window has nothing to evict still applies: a
        // redundant insert. The loop driver skips the summarizer call
        // entirely when `has_eviction_pending()` is false, so this path
        // only fires on direct API use (or a stale-plan bypass).
        let mut manager = ContextManager::new(config());
        manager.commit_turn(&[Message::user("u1"), Message::assistant("a1")]);
        assert!(!manager.has_eviction_pending());

        let record = manager
            .apply_summary("redundant")
            .expect("apply succeeds with no eviction range");
        assert_eq!(record.tokens_before, 0);
        // The summary was prepended; nothing was evicted.
        assert_eq!(manager.history.len(), 3);
        assert_eq!(manager.history[0].content(), "redundant");
        assert_eq!(manager.sticky_count(), 1);
    }

    #[test]
    fn apply_summary_twice_without_request_applies_both() {
        // No pairing enforcement: two `apply_summary` calls in a row
        // (no `summarize_request` between them) both succeed — the
        // stale check is skipped because `request_plan` is None.
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }

        let first = manager
            .apply_summary("first")
            .expect("first apply succeeds");
        assert_eq!(first.text, "first");
        assert_eq!(manager.sticky_count(), 1);
        assert_eq!(manager.history[0].content(), "first");

        let second = manager
            .apply_summary("second")
            .expect("second apply succeeds without a new summarize_request");
        assert_eq!(second.text, "second");
        // Single-generation: the second summary replaced the first in
        // place.
        assert_eq!(manager.sticky_count(), 1);
        assert_eq!(manager.history[0].content(), "second");
        assert!(!manager.history.iter().any(|m| m.content() == "first"));
        assert_eq!(
            manager.compaction_summary().map(|r| r.text.as_str()),
            Some("second")
        );
    }

    #[test]
    fn has_eviction_pending_false_for_tiny_history() {
        let mut manager = ContextManager::new(config());
        manager.commit_turn(&[Message::user("u1"), Message::assistant("a1")]);
        assert!(!manager.has_eviction_pending());
        manager.commit_turn(&[Message::user("u2"), Message::assistant("a2")]);
        // Still within the window: nothing would be summarized.
        assert!(!manager.has_eviction_pending());
    }

    #[test]
    fn has_eviction_pending_true_when_window_has_eviction_range() {
        let mut manager = ContextManager::new(window_config(10_000, 2_000, 2));
        for i in 0..60 {
            manager.commit_turn(&turn(i, 400));
        }
        assert!(manager.has_eviction_pending());
        // The accessor reflects the same eviction range the
        // summarization prompt would cover: at least the oldest turn.
        let ranges = manager.window_plan().disappearing_ranges();
        assert!(!ranges.is_empty());
        assert!(ranges.iter().any(|(start, end)| end > start));
    }

    #[test]
    fn load_compaction_summary_without_sticky_count_is_not_pinned() {
        // A schema_version-2 file with a CompactionRecord summary
        // present but `sticky_count: 0`, whose FIRST message does NOT
        // match the record text: load succeeds with the record
        // restored, but the summary is NOT pinned — the window evicts
        // it on the next prepare. AC-25/D-8 repair only fires on an
        // exact content match; a mismatched first message is left
        // untouched (no guessing).
        let dir = temp_session_dir("summary-not-pinned");
        std::fs::create_dir_all(&dir).expect("mkdir failed");

        let mut history = vec![json!({
            "role": "user",
            "content": [{"type": "text", "text": "summary from a newer build"}],
            "tool_call_id": null,
            "tool_calls": [],
        })];
        for i in 0..60 {
            history.push(json!({
                "role": "user",
                "content": [{"type": "text", "text": format!("u{i}:{}", "a".repeat(400))}],
                "tool_call_id": null,
                "tool_calls": [],
            }));
            history.push(json!({
                "role": "assistant",
                "content": [{"type": "text", "text": format!("a{i}:{}", "b".repeat(400))}],
                "tool_call_id": null,
                "tool_calls": [],
            }));
        }
        let state = json!({
            "session_id": "sess-1",
            "created_at": "0",
            "history": history,
            "compaction_summary": {
                "text": "stale summary text",
                "tokens_before": 12_000,
                "tokens_after": 7,
                "created_at": "0",
                "attempt": 1,
                "image_refs": [],
            },
            "sticky_count": 0,
            "total_tokens": 12_000,
            "schema_version": 2,
        });
        std::fs::write(dir.join("sess-1.json"), state.to_string()).expect("write failed");

        // Same window shape as `window_config(10_000, 2_000, 2)`, but
        // pointed at the temp session dir the file was written to.
        let cfg = ContextConfig {
            reserved_tokens: 2_000,
            tail_turns: 2,
            ..ContextConfig::new(10_000, dir)
        };
        let mut loaded = ContextManager::load("sess-1", cfg).expect("load failed");
        // Load succeeds and restores the record, but the sticky count
        // stays 0 — the file's first message does not match the record
        // text, so the D-8 repair does not guess.
        assert_eq!(loaded.sticky_count(), 0);
        let record = loaded.compaction_summary().expect("record restored");
        assert_eq!(record.text, "stale summary text");

        // The mismatched first message is not pinned: the window
        // evicts it on the next prepare, even though
        // `compaction_summary()` still reports the record.
        loaded.prepare_messages("sys", &[]);
        assert!(
            !loaded
                .history
                .iter()
                .any(|m| m.content() == "summary from a newer build"),
            "unpinned first message must be evicted by the window"
        );
    }

    #[test]
    fn load_repins_summary_when_first_message_matches_record() {
        // AC-25/D-8: a schema_version-2 file whose first message's
        // content equals the compaction record's text but whose
        // `sticky_count` was not persisted (0) loads with the pin
        // repaired — the summary message is present, it just lost its
        // sticky flag. The repaired pin survives the window.
        let dir = temp_session_dir("summary-repin");
        std::fs::create_dir_all(&dir).expect("mkdir failed");

        let mut history = vec![json!({
            "role": "user",
            "content": [{"type": "text", "text": "stale summary text"}],
            "tool_call_id": null,
            "tool_calls": [],
        })];
        for i in 0..60 {
            history.push(json!({
                "role": "user",
                "content": [{"type": "text", "text": format!("u{i}:{}", "a".repeat(400))}],
                "tool_call_id": null,
                "tool_calls": [],
            }));
            history.push(json!({
                "role": "assistant",
                "content": [{"type": "text", "text": format!("a{i}:{}", "b".repeat(400))}],
                "tool_call_id": null,
                "tool_calls": [],
            }));
        }
        let state = json!({
            "session_id": "sess-1",
            "created_at": "0",
            "history": history,
            "compaction_summary": {
                "text": "stale summary text",
                "tokens_before": 12_000,
                "tokens_after": 7,
                "created_at": "0",
                "attempt": 1,
                "image_refs": [],
            },
            "sticky_count": 0,
            "total_tokens": 12_000,
            "schema_version": 2,
        });
        std::fs::write(dir.join("sess-1.json"), state.to_string()).expect("write failed");

        let cfg = ContextConfig {
            reserved_tokens: 2_000,
            tail_turns: 2,
            ..ContextConfig::new(10_000, dir)
        };
        let mut loaded = ContextManager::load("sess-1", cfg).expect("load failed");
        assert_eq!(
            loaded.sticky_count(),
            1,
            "matching summary message is re-pinned"
        );
        let record = loaded.compaction_summary().expect("record restored");
        assert_eq!(record.text, "stale summary text");

        // The repaired pin survives the window: repeated prepares do
        // not evict the summary message.
        loaded.prepare_messages("sys", &[]);
        assert!(
            loaded
                .history
                .iter()
                .any(|m| m.content() == "stale summary text"),
            "re-pinned summary must not be evicted by the window"
        );
        assert!(loaded.token_count() <= 10_000);
    }
}
