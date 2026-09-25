//! Client for the [Agent Client Protocol][acp], backing the assist panel.
//!
//! The panel is a fixed-width dock that [`EditorView`](crate::ui::EditorView) owns and
//! draws, the way it owns the completion popup. It is carved off the right of the
//! editor area *before* the view tree is laid out, so it displaces buffers rather than
//! covering them. The transcript is rendered markdown, not a buffer; `:assist-yank`
//! copies it into a scratch buffer when you want to search or yank it.
//!
//! The connection runs as a detached tokio task. Anything it needs to show the user
//! goes through [`job::dispatch_blocking`], which lands in the main event loop holding
//! `&mut Editor` and `&mut Compositor`.
//!
//! One rule governs the handlers registered in [`run`]: **they must never await user
//! input**. The SDK's dispatch loop is serial and global, so a handler that blocks
//! stalls the whole connection and no further `session/update` would render. Handlers
//! hand their work to the main loop and return immediately.
//!
//! [acp]: https://agentclientprotocol.com

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use agent_client_protocol::schema::v1::{
    CancelNotification, ClientCapabilities, ContentBlock, Diff, FileSystemCapabilities,
    InitializeRequest, NewSessionRequest, PermissionOption, PermissionOptionKind, PromptRequest,
    ReadTextFileRequest, ReadTextFileResponse, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, SelectedPermissionOutcome, SessionNotification, SessionUpdate,
    TextContent, ToolCallContent, WriteTextFileRequest, WriteTextFileResponse,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{AcpAgent, Agent, ConnectionTo, Error as AcpError, Responder};
use anyhow::{anyhow, bail, Context as _, Result};
use helix_core::diff::compare_ropes;
use helix_core::{Assoc, ChangeSet, Rope, Selection, Transaction};
use helix_view::document::SavePoint;
use helix_view::editor::Action;
use helix_view::graphics::Rect;
use helix_view::input::KeyCode;
use helix_view::{Document, DocumentId, Editor, ViewId};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tui::buffer::Buffer as Surface;
use tui::widgets::{Paragraph, Widget, Wrap};

use crate::compositor::{Component, Compositor, Context, Event, EventResult};
use crate::job;

/// A live assist session: one agent subprocess and the buffer showing its transcript.
pub struct AssistSession {
    /// Rendered transcript, as markdown.
    transcript: String,
    /// Lines scrolled back from the newest output. Zero pins the panel to the bottom.
    scroll: u16,
    /// Outbound queue to the connection task.
    tx: UnboundedSender<Request>,
    /// Whether a prompt turn is in flight.
    busy: bool,
    /// The edit currently awaiting a verdict, if any.
    review: Option<PendingReview>,
}

/// One proposed edit, already applied to its buffer.
struct AppliedEdit {
    doc: DocumentId,
    view: ViewId,
    /// State to restore on reject.
    savepoint: Arc<SavePoint>,
}

/// A proposal awaiting the user's verdict.
///
/// The edits are already in their documents when this exists; the savepoints are how
/// we take them back out. Nothing is committed to history until the user accepts, so
/// rejecting leaves no undo residue.
struct PendingReview {
    /// `None` once answered. Held in an `Option` so [`Drop`] can answer for us.
    responder: Option<Responder<RequestPermissionResponse>>,
    /// Options as the agent supplied them; we pick by kind, never by position.
    options: Vec<PermissionOption>,
    /// Every edit in this proposal. A tool call may touch several files, and all of
    /// them stand or fall together.
    edits: Vec<AppliedEdit>,
    /// True once the user took over to hand-correct the edit.
    editing: bool,
}

impl Drop for PendingReview {
    fn drop(&mut self) {
        // An unanswered request hangs the agent forever with no error anywhere, so a
        // review that is dropped (panel closed, session replaced) still gets a verdict.
        if let Some(responder) = self.responder.take() {
            let _ = responder.respond(RequestPermissionResponse::new(
                RequestPermissionOutcome::Cancelled,
            ));
        }
    }
}

/// What the user decided about a proposed edit.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Accept,
    Reject,
    /// Dismiss the gate and let the user fix the edit by hand.
    Edit,
}

/// Work sent from the editor to the connection task.
enum Request {
    Prompt(String),
    Cancel,
}

impl AssistSession {
    /// The transcript so far, for `:assist-yank`.
    pub fn transcript(&self) -> &str {
        &self.transcript
    }

    pub fn is_busy(&self) -> bool {
        self.busy
    }

    /// Scroll the panel back (positive) or toward the newest output (negative).
    pub fn scroll(&mut self, lines: i16) {
        self.scroll = self.scroll.saturating_add_signed(lines);
    }

    /// Send a user message. Ignored while a turn is already in flight.
    pub fn prompt(&mut self, text: String) {
        if self.busy {
            return;
        }
        self.busy = true;
        let _ = self.tx.send(Request::Prompt(text));
    }

    /// Ask the agent to abandon the current turn.
    pub fn cancel(&self) {
        let _ = self.tx.send(Request::Cancel);
    }

    /// Whether the connection task is still alive.
    pub fn is_connected(&self) -> bool {
        !self.tx.is_closed()
    }
}

/// Start an agent and return the session for `EditorView` to hold and draw.
///
/// `carry_over` is the transcript of a session whose agent died, so restarting keeps
/// the history on screen.
pub fn open(editor: &mut Editor, carry_over: Option<String>) -> Result<AssistSession> {
    let config = editor.config();
    if !config.assist.enable {
        bail!("assist is disabled (`editor.assist.enable`)");
    }
    let command = config
        .assist
        .command
        .clone()
        .context("no assist agent configured; set `editor.assist.command`")?;
    drop(config);

    let cwd = helix_stdx::env::current_working_dir();

    let mut transcript = carry_over.unwrap_or_default();
    if !transcript.is_empty() {
        transcript.push_str("\n---\n_restarting agent_\n");
    }

    let (tx, rx) = unbounded_channel();
    tokio::spawn(async move {
        let result = run(command, cwd, rx).await;
        let message = match result {
            Ok(()) => "assist: agent disconnected".to_string(),
            Err(err) => format!("assist: {err:#}"),
        };
        job::dispatch_blocking(move |editor, _compositor| {
            editor.set_error(message);
        });
    });

    Ok(AssistSession {
        transcript,
        scroll: 0,
        tx,
        busy: false,
        review: None,
    })
}

/// Entry point for both the `assist_open` keybinding and `:assist`.
///
/// Resumes a hand-corrected proposal if one is waiting, otherwise makes sure a session
/// is running and asks for a message.
pub fn open_or_prompt(editor: &mut Editor, compositor: &mut Compositor) {
    if resume_review(compositor) {
        return;
    }

    let session = session_mut(compositor);
    let running = session
        .as_ref()
        .is_some_and(|session| session.is_connected());
    let stale_transcript = session.map(|session| session.transcript.clone());
    if !running {
        match open(editor, stale_transcript) {
            Ok(session) => {
                if let Some(view) = compositor.find::<crate::ui::EditorView>() {
                    view.assist = Some(session);
                }
            }
            Err(err) => {
                editor.set_error(format!("{err:#}"));
                return;
            }
        }
    }

    prompt();
}

/// Abandon the current turn: tell the agent, and take back any unreviewed edit.
pub fn cancel(editor: &mut Editor, compositor: &mut Compositor) {
    let Some(session) = session_mut(compositor) else {
        editor.set_error("assist: no session");
        return;
    };
    session.cancel();

    // Dropping the review answers the outstanding permission request with `Cancelled`,
    // which the protocol requires of a cancelled turn.
    if let Some(review) = session.review.take() {
        revert_all(editor, &review.edits);
    }
    compositor.remove_type::<ReviewGate>();
    editor.set_status("assist: cancelled");
}

/// Show the prompt used to type a message to the agent.
///
/// Queued through the job channel rather than pushed inline, because opening the panel
/// moves focus and the `DocumentFocusLost` hook enqueues a `remove_type::<Prompt>()`.
/// Pushing inline would put the prompt up just in time for that removal to take it
/// back down; going through the same FIFO puts us after it.
pub fn prompt() {
    job::dispatch_blocking(|_editor, compositor| {
        let prompt = crate::ui::Prompt::new(
            "assist: ".into(),
            Some('a'),
            crate::ui::completers::none,
            |_cx, input, event| {
                if event != crate::ui::PromptEvent::Validate || input.trim().is_empty() {
                    return;
                }
                submit_prompt(input.to_string());
            },
        );
        compositor.push(Box::new(prompt));
    });
}

/// Carve the panel out of the editor area, or `None` when the window cannot spare it.
///
/// Called before the view tree is laid out, so whatever this returns is taken away
/// from the buffers rather than drawn on top of them.
pub fn panel_area(area: Rect, width: u16) -> Option<Rect> {
    let width = width.min(area.width / 2);
    // Narrow windows shrink the panel rather than losing it, but below a readable
    // width, or once the code would be squeezed under 30 columns, drop it entirely.
    if width < 24 || area.width.saturating_sub(width) < 30 {
        return None;
    }
    Some(Rect::new(area.right() - width, area.y, width, area.height))
}

impl AssistSession {
    /// Draw the transcript. Markdown so headings and fenced code come out highlighted.
    pub fn render_panel(&mut self, area: Rect, surface: &mut Surface, cx: &mut Context) {
        let theme = &cx.editor.theme;
        surface.clear_with(area, theme.get("ui.background"));

        // Vertical rule marking the panel off from the buffers.
        let rule = theme.get("ui.window");
        for y in area.top()..area.bottom() {
            surface.set_string(area.x, y, "\u{2502}", rule);
        }

        let inner = area.clip_left(2).clip_right(1);
        if inner.width == 0 || inner.height == 0 {
            return;
        }

        let status = if self.review.is_some() {
            "review"
        } else if self.busy {
            "working"
        } else {
            "idle"
        };
        let header = theme.get("ui.statusline");
        surface.clear_with(Rect::new(inner.x, inner.y, inner.width, 1), header);
        surface.set_stringn(
            inner.x,
            inner.y,
            &format!(" assist \u{b7} {status}"),
            inner.width as usize,
            header,
        );

        let body = inner.clip_top(2);
        if body.height == 0 {
            return;
        }

        let markdown =
            crate::ui::Markdown::new(self.transcript.clone(), cx.editor.syn_loader.clone());
        let text = markdown.parse(Some(theme));
        let (_, rendered) = crate::ui::text::required_size(&text, body.width);

        // `scroll` counts lines back from the newest output, so zero sits at the bottom
        // and the panel follows a streaming reply without any work.
        let max = rendered.saturating_sub(body.height);
        self.scroll = self.scroll.min(max);
        let offset = max - self.scroll;

        Paragraph::new(&text)
            .wrap(Wrap { trim: false })
            .scroll((offset, 0))
            .render(body, surface);
    }
}

/// Copy the transcript into an ordinary scratch buffer, where search and yank work.
pub fn yank_transcript(editor: &mut Editor, compositor: &mut Compositor) {
    let Some(text) = session_mut(compositor).map(|session| session.transcript.clone()) else {
        editor.set_error("assist: no session");
        return;
    };

    let doc_id = editor.new_file(Action::HorizontalSplit);
    let view_id = editor.tree.focus;
    let doc = doc_mut!(editor, &doc_id);
    doc.ensure_view_init(view_id);
    let transaction = Transaction::change(
        doc.text(),
        std::iter::once((0, 0, Some(text.as_str().into()))),
    );
    doc.apply(&transaction, view_id);
    doc.set_selection(view_id, Selection::point(0));
}

/// Queue a user message for the running session.
///
/// Routed through the job queue so callers that only hold a `compositor::Context`
/// (a [`Prompt`](crate::ui::Prompt) callback, say) can still reach the session.
pub fn submit_prompt(text: String) {
    job::dispatch_blocking(move |editor, compositor| match session_mut(compositor) {
        Some(session) => session.prompt(text),
        None => editor.set_error("assist: no session"),
    });
}

/// Drive one agent connection for the lifetime of the session.
async fn run(command: String, cwd: PathBuf, mut rx: UnboundedReceiver<Request>) -> Result<()> {
    let agent = AcpAgent::from_str(&command)
        .map_err(|err| anyhow!("could not launch agent `{command}`: {err}"))?;

    // Both default to false; an agent that sees them unset will never ask us for file
    // contents and will read stale bytes off disk instead.
    let capabilities = ClientCapabilities::new()
        .fs(FileSystemCapabilities::new()
            .read_text_file(true)
            .write_text_file(true))
        .terminal(false);

    agent_client_protocol::Client
        .builder()
        .name("helix")
        .on_receive_notification(
            async move |notification: SessionNotification, _cx| {
                job::dispatch_blocking(move |_editor, compositor| {
                    apply_update(compositor, notification.update);
                });
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |request: RequestPermissionRequest, responder, _cx| {
                // Hand the request *and its responder* to the main loop. Waiting for a
                // keypress here would block the dispatch loop and freeze the session.
                job::dispatch_blocking(move |editor, compositor| {
                    begin_review(editor, compositor, request, responder);
                });
                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: ReadTextFileRequest, responder, _cx| {
                job::dispatch_blocking(move |editor, _compositor| {
                    let _ = responder.respond_with_result(read_text_file(editor, &request));
                });
                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: WriteTextFileRequest, responder, _cx| {
                job::dispatch_blocking(move |editor, _compositor| {
                    let _ = responder.respond_with_result(write_text_file(editor, &request));
                });
                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(agent, move |cx: ConnectionTo<Agent>| async move {
            cx.send_request(
                InitializeRequest::new(ProtocolVersion::V1).client_capabilities(capabilities),
            )
            .block_task()
            .await?;

            let session = cx
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            let session_id = session.session_id;

            while let Some(request) = rx.recv().await {
                let text = match request {
                    // A cancel with no turn in flight is a no-op.
                    Request::Cancel => continue,
                    Request::Prompt(text) => text,
                };

                let echo = text.clone();
                job::dispatch_blocking(move |_editor, compositor| {
                    append(compositor, &format!("\n### you\n\n{echo}\n"));
                });

                let prompt = cx
                    .send_request(PromptRequest::new(
                        session_id.clone(),
                        vec![ContentBlock::Text(TextContent::new(text))],
                    ))
                    .block_task();
                tokio::pin!(prompt);

                // Keep servicing the queue while the turn runs, so a cancel can reach
                // the agent. Note we await the prompt to completion even after
                // cancelling: dropping the handle would send `$/cancel_request`, which
                // is request cancellation, not turn cancellation.
                let response = loop {
                    tokio::select! {
                        response = &mut prompt => break response?,
                        Some(request) = rx.recv() => {
                            if matches!(request, Request::Cancel) {
                                cx.send_notification(CancelNotification::new(session_id.clone()))?;
                            }
                        }
                    }
                };

                let stop = response.stop_reason;
                job::dispatch_blocking(move |_editor, compositor| {
                    if let Some(session) = session_mut(compositor) {
                        session.busy = false;
                    }
                    append(compositor, &format!("\n_({stop:?})_\n"));
                });
            }

            Ok(())
        })
        .await?;

    Ok(())
}

/// Render one `session/update` into the transcript.
fn apply_update(compositor: &mut Compositor, update: SessionUpdate) {
    match update {
        SessionUpdate::AgentMessageChunk(chunk) => {
            if let Some(text) = content_text(&chunk.content) {
                append(compositor, &text);
            }
        }
        SessionUpdate::AgentThoughtChunk(chunk) => {
            if let Some(text) = content_text(&chunk.content) {
                append(compositor, &format!("\n> {text}\n"));
            }
        }
        SessionUpdate::Plan(plan) => {
            let mut rendered = String::from("\n### plan\n\n");
            for entry in &plan.entries {
                rendered.push_str(&format!("- [{:?}] {}\n", entry.status, entry.content));
            }
            append(compositor, &rendered);
        }
        SessionUpdate::ToolCall(call) => {
            append(compositor, &format!("\n`{}`\n", call.title));
        }
        // Tool-call updates, mode changes, usage and everything added to the protocol
        // after this was written are not surfaced yet. The enum is `#[non_exhaustive]`.
        _ => {}
    }
}

fn content_text(content: &ContentBlock) -> Option<String> {
    match content {
        ContentBlock::Text(text) => Some(text.text.clone()),
        _ => None,
    }
}

/// Append to the transcript, snapping the panel back to the newest output.
fn append(compositor: &mut Compositor, text: &str) {
    if let Some(session) = session_mut(compositor) {
        session.transcript.push_str(text);
        session.scroll = 0;
    }
}

fn view_showing(editor: &Editor, doc: DocumentId) -> Option<ViewId> {
    editor
        .tree
        .views()
        .find(|(view, _)| view.doc == doc)
        .map(|(view, _)| view.id)
}

/// Reach the session stored on `EditorView` from a job callback.
fn session_mut(compositor: &mut Compositor) -> Option<&mut AssistSession> {
    compositor
        .find::<crate::ui::EditorView>()
        .and_then(|view| view.assist.as_mut())
}

/// Apply a proposed edit to its buffer and put the review gate up.
///
/// The edit lands in the real document so the user reviews it in place, with live LSP
/// diagnostics on the result — a bad proposal lights up before it is accepted.
fn begin_review(
    editor: &mut Editor,
    compositor: &mut Compositor,
    request: RequestPermissionRequest,
    responder: Responder<RequestPermissionResponse>,
) {
    let title = request
        .tool_call
        .fields
        .title
        .clone()
        .unwrap_or_else(|| "proposed edit".to_string());

    let diffs: Vec<Diff> = request
        .tool_call
        .fields
        .content
        .as_ref()
        .map(|content| {
            content
                .iter()
                .filter_map(|item| match item {
                    ToolCallContent::Diff(diff) => Some(diff.clone()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();

    // Anything we cannot show as an in-buffer edit is declined rather than guessed at.
    if diffs.is_empty() {
        decline(
            compositor,
            responder,
            &format!("{title}: nothing to review"),
        );
        return;
    }

    // All-or-nothing: if any file in the proposal fails to apply, back out the ones
    // that already did rather than leaving a half-applied edit under review.
    let mut edits = Vec::with_capacity(diffs.len());
    for diff in &diffs {
        match apply_proposal(editor, diff) {
            Ok(edit) => edits.push(edit),
            Err(err) => {
                revert_all(editor, &edits);
                decline(compositor, responder, &format!("{title}: {err:#}"));
                return;
            }
        }
    }

    let review = PendingReview {
        responder: Some(responder),
        options: request.options,
        edits,
        editing: false,
    };

    // If the session vanished, dropping `review` answers the agent for us.
    let Some(session) = session_mut(compositor) else {
        return;
    };
    session.review = Some(review);

    let scope = if diffs.len() == 1 {
        String::new()
    } else {
        format!(" ({} files)", diffs.len())
    };
    append(compositor, &format!("\n_reviewing_ `{title}`{scope}\n"));
    compositor.push(Box::new(ReviewGate {
        title: format!("{title}{scope}"),
    }));
}

/// Answer the agent without touching any buffer.
fn decline(
    compositor: &mut Compositor,
    responder: Responder<RequestPermissionResponse>,
    reason: &str,
) {
    append(compositor, &format!("\n_(declined: {reason})_\n"));
    let _ = responder.respond(RequestPermissionResponse::new(
        RequestPermissionOutcome::Cancelled,
    ));
}

/// Open the target file, savepoint it, and apply the proposal as a minimal transaction.
fn apply_proposal(editor: &mut Editor, diff: &Diff) -> Result<AppliedEdit> {
    let doc_id = editor
        .open(&diff.path, Action::Replace)
        .map_err(|err| anyhow!("could not open {}: {err}", diff.path.display()))?;
    let view_id = editor.tree.focus;

    let savepoint = {
        let (view, doc) = current!(editor);
        doc.ensure_view_init(view.id);
        // Taken before applying. `Document::apply` composes inverses into live
        // savepoints, so this survives the user hand-editing during review — which a
        // pre-captured `Transaction::invert` would not.
        let savepoint = doc.savepoint(view);

        let transaction = proposal_transaction(doc.text(), diff)?;
        let span = changed_span(transaction.changes());

        doc.apply(&transaction, view.id);
        if let Some((from, to)) = span {
            // Clamp: a mapped span must never outrun the text it is selecting.
            let last = doc.text().len_chars();
            doc.set_selection(view.id, Selection::single(from.min(last), to.min(last)));
        }
        savepoint
    };

    editor.ensure_cursor_in_view(view_id);
    Ok(AppliedEdit {
        doc: doc_id,
        view: view_id,
        savepoint,
    })
}

/// Build the transaction for a proposed diff.
///
/// ACP's `Diff` does not say whether `new_text` is the whole file or just the changed
/// region, and real agents differ: Claude's adapter sends a hunk together with the
/// matching `old_text`, while a whole-file proposal arrives with `old_text` equal to
/// the current contents, or absent for a new file. Guessing wrong replaces the entire
/// buffer with a fragment, so key off `old_text` and refuse when it does not fit rather
/// than destroy the buffer.
fn proposal_transaction(text: &Rope, diff: &Diff) -> Result<Transaction> {
    let Some(old) = diff.old_text.as_deref().filter(|old| !old.is_empty()) else {
        // A new file, or a whole-file replacement.
        return Ok(compare_ropes(text, &Rope::from(diff.new_text.as_str())));
    };

    let current = text.to_string();
    if current == old {
        // Whole-file form: diff it so the selection lands on what actually changed.
        return Ok(compare_ropes(text, &Rope::from(diff.new_text.as_str())));
    }

    let mut matches = current.match_indices(old);
    let (byte, _) = matches
        .next()
        .context("the proposed edit does not match the buffer")?;
    if matches.next().is_some() {
        bail!("the proposed edit matches the buffer in more than one place");
    }

    let from = text.byte_to_char(byte);
    let to = from + old.chars().count();
    Ok(Transaction::change(
        text,
        std::iter::once((from, to, Some(diff.new_text.as_str().into()))),
    ))
}

/// The span the change set touches, in coordinates of the *new* text.
fn changed_span(changes: &ChangeSet) -> Option<(usize, usize)> {
    let mut span: Option<(usize, usize)> = None;
    for (from, _to, insert) in changes.changes_iter() {
        // `Assoc::Before` keeps the mapped position *ahead* of the inserted text, so
        // adding the insert length lands on its end. `Assoc::After` would already be
        // past it and the span would overrun the document.
        let start = changes.map_pos(from, Assoc::Before);
        let end = start + insert.as_ref().map_or(0, |text| text.chars().count());
        span = Some(match span {
            None => (start, end),
            Some((lo, hi)) => (lo.min(start), hi.max(end)),
        });
    }
    span
}

/// Re-open the gate after the user hand-corrected a proposal. Returns whether it did.
pub fn resume_review(compositor: &mut Compositor) -> bool {
    let resumed = match session_mut(compositor).and_then(|session| session.review.as_mut()) {
        Some(review) if review.editing => {
            review.editing = false;
            true
        }
        _ => false,
    };
    if resumed {
        compositor.push(Box::new(ReviewGate {
            title: "edited proposal".to_string(),
        }));
    }
    resumed
}

/// Act on the user's verdict and answer the agent.
fn resolve(compositor: &mut Compositor, editor: &mut Editor, verdict: Verdict) {
    let Some(session) = session_mut(compositor) else {
        return;
    };
    let Some(mut review) = session.review.take() else {
        return;
    };

    if verdict == Verdict::Edit {
        // Leave the edit in place, unanswered, and hand the buffer back to the user.
        review.editing = true;
        if let Some(session) = session_mut(compositor) {
            session.review = Some(review);
        }
        editor.set_status("assist: editing proposal — space+A to resume review");
        return;
    }

    let note = match verdict {
        Verdict::Accept => {
            // Commit exactly one undo revision per touched file.
            for edit in &review.edits {
                if !editor.tree.contains(edit.view) || !editor.documents.contains_key(&edit.doc) {
                    continue;
                }
                let view = view_mut!(editor, edit.view);
                let doc = doc_mut!(editor, &edit.doc);
                doc.append_changes_to_history(view);
            }
            "accepted"
        }
        Verdict::Reject => {
            revert_all(editor, &review.edits);
            "rejected"
        }
        Verdict::Edit => unreachable!("handled above"),
    };

    if let Some(responder) = review.responder.take() {
        let outcome = pick_option(&review.options, verdict == Verdict::Accept)
            .map(|id| RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(id)))
            .unwrap_or(RequestPermissionOutcome::Cancelled);
        let _ = responder.respond(RequestPermissionResponse::new(outcome));
    }

    append(compositor, &format!("\n_({note})_\n"));
}

/// Pick an option by kind. Options are agent-supplied, so never index by position.
fn pick_option(
    options: &[PermissionOption],
    allow: bool,
) -> Option<agent_client_protocol::schema::v1::PermissionOptionId> {
    let matches = |kind: &PermissionOptionKind| match kind {
        PermissionOptionKind::AllowOnce | PermissionOptionKind::AllowAlways => allow,
        PermissionOptionKind::RejectOnce | PermissionOptionKind::RejectAlways => !allow,
        _ => false,
    };
    // Prefer the "once" forms: a keypress should not silently grant standing consent.
    let once = |kind: &PermissionOptionKind| {
        matches!(
            kind,
            PermissionOptionKind::AllowOnce | PermissionOptionKind::RejectOnce
        )
    };
    options
        .iter()
        .find(|option| matches(&option.kind) && once(&option.kind))
        .or_else(|| options.iter().find(|option| matches(&option.kind)))
        .map(|option| option.option_id.clone())
}

/// One-line bar asking for a verdict on the proposal sitting in the buffer.
struct ReviewGate {
    title: String,
}

impl Component for ReviewGate {
    fn render(&mut self, area: Rect, surface: &mut Surface, cx: &mut Context) {
        let style = cx.editor.theme.get("ui.statusline");
        let row = area.y + area.height.saturating_sub(1);
        let hint = format!(" {}   [a]ccept  [r]eject  [e]dit ", self.title);
        let bar = Rect::new(area.x, row, area.width, 1);
        surface.clear_with(bar, style);
        surface.set_string(area.x, row, &hint, style);
    }

    fn handle_event(&mut self, event: &Event, _cx: &mut Context) -> EventResult {
        let Event::Key(key) = event else {
            return EventResult::Ignored(None);
        };
        let verdict = match key.code {
            KeyCode::Char('a') => Verdict::Accept,
            KeyCode::Char('r') => Verdict::Reject,
            KeyCode::Char('e') | KeyCode::Esc => Verdict::Edit,
            // Swallow anything else: a stray key must not edit a buffer that is holding
            // an unreviewed proposal.
            _ => return EventResult::Consumed(None),
        };
        EventResult::Consumed(Some(Box::new(
            move |compositor: &mut Compositor, cx: &mut Context| {
                compositor.remove_type::<ReviewGate>();
                resolve(compositor, cx.editor, verdict);
            },
        )))
    }
}

/// Serve `fs/read_text_file` from the open buffer when there is one.
///
/// This is the reason ACP has client-side filesystem methods at all: an agent that
/// reads the file itself sees what is on disk, which is stale the moment the user has
/// unsaved edits — or the moment we apply a proposal that is still under review.
fn read_text_file(
    editor: &Editor,
    request: &ReadTextFileRequest,
) -> std::result::Result<ReadTextFileResponse, AcpError> {
    let text = match open_document(editor, &request.path) {
        Some(doc) => doc.text().to_string(),
        None => std::fs::read_to_string(&request.path)
            .map_err(|_| AcpError::resource_not_found(Some(request.path.display().to_string())))?,
    };

    Ok(ReadTextFileResponse::new(slice_lines(
        &text,
        request.line,
        request.limit,
    )))
}

/// Serve `fs/write_text_file`.
///
/// An open buffer is edited in place and left unsaved, so the change is visible, and
/// undoable, before it ever reaches disk. Files that are not open are written through,
/// creating them if needed as the protocol requires.
fn write_text_file(
    editor: &mut Editor,
    request: &WriteTextFileRequest,
) -> std::result::Result<WriteTextFileResponse, AcpError> {
    let doc_id = open_document(editor, &request.path).map(|doc| doc.id());

    match doc_id {
        Some(doc_id) => {
            let view_id = view_showing(editor, doc_id).unwrap_or(editor.tree.focus);
            let doc = editor
                .documents
                .get_mut(&doc_id)
                .ok_or_else(AcpError::internal_error)?;
            doc.ensure_view_init(view_id);
            let after = Rope::from(request.content.as_str());
            let transaction = compare_ropes(doc.text(), &after);
            doc.apply(&transaction, view_id);
        }
        None => {
            if let Some(parent) = request.path.parent() {
                std::fs::create_dir_all(parent).map_err(|_| AcpError::internal_error())?;
            }
            std::fs::write(&request.path, &request.content)
                .map_err(|_| AcpError::internal_error())?;
        }
    }

    Ok(WriteTextFileResponse::new())
}

/// The open document for `path`, if the editor has one.
fn open_document<'a>(editor: &'a Editor, path: &Path) -> Option<&'a Document> {
    let path = helix_stdx::path::normalize(path);
    editor
        .documents
        .values()
        .find(|doc| doc.path().is_some_and(|open| *open == path))
}

/// Take a whole proposal back out of its buffers.
///
/// Each restore reverts everything since its savepoint, which during a review is
/// exactly that file's edit plus any hand-corrections made to it. Reverse order so
/// two edits to the same file unwind correctly.
fn revert_all(editor: &mut Editor, edits: &[AppliedEdit]) {
    for edit in edits.iter().rev() {
        if !editor.tree.contains(edit.view) || !editor.documents.contains_key(&edit.doc) {
            continue;
        }
        let view = view_mut!(editor, edit.view);
        let doc = doc_mut!(editor, &edit.doc);
        doc.restore(view, &edit.savepoint, true);
    }
}

/// Apply `fs/read_text_file`'s `line`/`limit` window. `line` is 1-based and inclusive.
fn slice_lines(text: &str, line: Option<u32>, limit: Option<u32>) -> String {
    if line.is_none() && limit.is_none() {
        return text.to_string();
    }
    let skip = line.unwrap_or(1).saturating_sub(1) as usize;
    let mut lines: Vec<&str> = text.lines().skip(skip).collect();
    if let Some(limit) = limit {
        lines.truncate(limit as usize);
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute the span for an edit and assert it addresses the *new* text.
    fn span_of(before: &str, after: &str) -> Option<(usize, usize)> {
        let before = Rope::from(before);
        let after = Rope::from(after);
        let transaction = compare_ropes(&before, &after);
        let span = changed_span(transaction.changes());
        if let Some((from, to)) = span {
            assert!(from <= to, "span {from}..{to} is inverted");
            assert!(
                to <= after.len_chars(),
                "span end {to} outruns the new text ({} chars)",
                after.len_chars()
            );
        }
        span
    }

    #[test]
    fn changed_span_selects_the_inserted_text() {
        let before = "fn main() {}\n";
        let after = "fn main() {}\nfn other() {}\n";
        let (from, to) = span_of(before, after).expect("an insertion has a span");
        let selected: String = after.chars().skip(from).take(to - from).collect();
        assert!(
            selected.contains("fn other"),
            "expected the new function to be selected, got {selected:?}"
        );
    }

    /// Regression: mapping the change start with `Assoc::After` put the position past
    /// the insertion, so `start + len` ran off the end of the rope and panicked inside
    /// `helix_core::graphemes`.
    #[test]
    fn changed_span_never_outruns_the_new_text() {
        let cases = [
            ("", "hello"),
            ("hello", ""),
            (
                "fn main() {\n    println!(\"hi\");\n}\n",
                "fn main() {\n    greet();\n}\n\nfn greet() {\n    println!(\"hello\");\n}\n",
            ),
            ("a\nb\nc\n", "a\nc\n"),
            ("one", "one"),
            ("x\n", "y\nz\nw\n"),
        ];
        for (before, after) in cases {
            span_of(before, after);
        }
    }

    #[test]
    fn unchanged_text_has_no_span() {
        assert_eq!(span_of("same\n", "same\n"), None);
    }

    fn diff_of(path: &str, old: Option<&str>, new: &str) -> Diff {
        let mut diff = Diff::new(std::path::PathBuf::from(path), new.to_string());
        diff.old_text = old.map(str::to_string);
        diff
    }

    fn applied(before: &str, old: Option<&str>, new: &str) -> Result<String> {
        let text = Rope::from(before);
        let transaction = proposal_transaction(&text, &diff_of("/tmp/x.rs", old, new))?;
        let mut after = text.clone();
        assert!(transaction.apply(&mut after), "transaction should apply");
        Ok(after.to_string())
    }

    /// Claude's adapter sends only the changed region. Treating that as whole-file
    /// content replaced the entire buffer with a single line.
    #[test]
    fn a_hunk_replaces_only_its_own_span() {
        let before = "def greet(name):\n    print(name)\n\ngreet(\"world\")\n";
        let after = applied(before, Some("def greet(name):"), "def say_hello(name):").unwrap();
        assert_eq!(
            "def say_hello(name):\n    print(name)\n\ngreet(\"world\")\n", after,
            "the rest of the file must survive"
        );
    }

    #[test]
    fn whole_file_old_text_still_works() {
        let before = "one\ntwo\n";
        let after = applied(before, Some(before), "one\nTWO\n").unwrap();
        assert_eq!("one\nTWO\n", after);
    }

    #[test]
    fn absent_old_text_is_a_whole_file_replacement() {
        let after = applied("stale\n", None, "fresh\n").unwrap();
        assert_eq!("fresh\n", after);
    }

    /// Better to decline than to clobber a buffer the agent has stale knowledge of.
    #[test]
    fn a_hunk_that_does_not_match_is_refused() {
        let err = applied("actual contents\n", Some("something else"), "new").unwrap_err();
        assert!(err.to_string().contains("does not match"), "{err}");
    }

    #[test]
    fn an_ambiguous_hunk_is_refused() {
        let err = applied("dup\ndup\n", Some("dup"), "changed").unwrap_err();
        assert!(err.to_string().contains("more than one place"), "{err}");
    }

    fn options() -> Vec<PermissionOption> {
        vec![
            PermissionOption::new("aa", "Always allow", PermissionOptionKind::AllowAlways),
            PermissionOption::new("a1", "Allow once", PermissionOptionKind::AllowOnce),
            PermissionOption::new("r1", "Reject once", PermissionOptionKind::RejectOnce),
            PermissionOption::new("ra", "Always reject", PermissionOptionKind::RejectAlways),
        ]
    }

    /// A single keypress must not grant standing consent.
    #[test]
    fn pick_option_prefers_the_once_forms() {
        assert_eq!(&*pick_option(&options(), true).unwrap().0, "a1");
        assert_eq!(&*pick_option(&options(), false).unwrap().0, "r1");
    }

    #[test]
    fn pick_option_falls_back_to_always_when_that_is_all_there_is() {
        let only_always = vec![
            PermissionOption::new("aa", "Always allow", PermissionOptionKind::AllowAlways),
            PermissionOption::new("ra", "Always reject", PermissionOptionKind::RejectAlways),
        ];
        assert_eq!(&*pick_option(&only_always, true).unwrap().0, "aa");
        assert_eq!(&*pick_option(&only_always, false).unwrap().0, "ra");
    }

    #[test]
    fn pick_option_reports_no_usable_option() {
        assert!(pick_option(&[], true).is_none());
    }

    #[test]
    fn slice_lines_windows_from_a_one_based_line() {
        let text = "one\ntwo\nthree\nfour\n";
        assert_eq!(slice_lines(text, None, None), text);
        assert_eq!(slice_lines(text, Some(2), None), "two\nthree\nfour");
        assert_eq!(slice_lines(text, Some(2), Some(2)), "two\nthree");
        assert_eq!(slice_lines(text, Some(1), Some(1)), "one");
        // A window past the end is empty, not an error.
        assert_eq!(slice_lines(text, Some(99), Some(2)), "");
    }
}
