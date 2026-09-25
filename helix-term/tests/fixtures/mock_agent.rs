//! Deterministic ACP agent used by the assist integration tests.
//!
//! Answers `initialize` and `session/new`, streams a few `session/update`s for any
//! prompt, then proposes one edit to the file named by `HX_MOCK_AGENT_TARGET` and
//! reports the verdict back. Built only with the `integration` feature.

use agent_client_protocol::schema::v1::{
    AgentCapabilities, ContentBlock, ContentChunk, Diff, InitializeRequest, InitializeResponse,
    NewSessionRequest, NewSessionResponse, PermissionOption, PermissionOptionKind, Plan, PlanEntry,
    PlanEntryPriority, PlanEntryStatus, PromptRequest, PromptResponse, RequestPermissionRequest,
    SessionNotification, SessionUpdate, StopReason, TextContent, ToolCallContent, ToolCallUpdate,
    ToolCallUpdateFields, ToolKind,
};
use agent_client_protocol::{Agent, Client, ConnectionTo, Result, Stdio};
use std::path::PathBuf;
use std::sync::OnceLock;

/// Working directory from `session/new`, so the proposed edit can name a real file.
static CWD: OnceLock<PathBuf> = OnceLock::new();

const REFACTORED: &str = r#"fn main() {
    greet("world");
}

fn greet(name: &str) {
    println!("hello, {name}");
}
"#;

#[tokio::main]
async fn main() -> Result<()> {
    Agent
        .builder()
        .name("mock-agent")
        .on_receive_request(
            async move |request: InitializeRequest, responder, _cx| {
                responder.respond(
                    InitializeResponse::new(request.protocol_version)
                        .agent_capabilities(AgentCapabilities::new()),
                )
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: NewSessionRequest, responder, _cx| {
                let _ = CWD.set(request.cwd.clone());
                responder.respond(NewSessionResponse::new("mock-session"))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: PromptRequest, responder, cx: ConnectionTo<Client>| {
                // Never do the work inline: the dispatch loop is serial, so awaiting
                // here would stall every other message on the connection.
                let conn = cx.clone();
                cx.spawn(async move {
                    let session = request.session_id.clone();
                    let say = |update| {
                        conn.send_notification(SessionNotification::new(session.clone(), update))
                    };

                    say(SessionUpdate::AgentThoughtChunk(ContentChunk::new(
                        ContentBlock::Text(TextContent::new("Considering the request.")),
                    )))?;

                    say(SessionUpdate::Plan(Plan::new(vec![
                        PlanEntry::new(
                            "Read the file",
                            PlanEntryPriority::High,
                            PlanEntryStatus::Completed,
                        ),
                        PlanEntry::new(
                            "Propose an edit",
                            PlanEntryPriority::Medium,
                            PlanEntryStatus::Pending,
                        ),
                    ])))?;

                    for chunk in ["Here is ", "a streamed ", "reply.\n"] {
                        say(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                            ContentBlock::Text(TextContent::new(chunk)),
                        )))?;
                    }

                    // Propose an edit and wait for the editor's verdict. Safe to block
                    // here: this is a spawned task, not a dispatch handler.
                    let path = match std::env::var("HX_MOCK_AGENT_TARGET") {
                        Ok(target) => PathBuf::from(target),
                        Err(_) => CWD.get().cloned().unwrap_or_default().join("demo.rs"),
                    };
                    let tool_call = ToolCallUpdate::new(
                        "edit-1",
                        ToolCallUpdateFields::new()
                            .title("Extract a greet function")
                            .kind(ToolKind::Edit)
                            .content(vec![ToolCallContent::Diff(Diff::new(path, REFACTORED))]),
                    );
                    let decision = conn
                        .send_request(RequestPermissionRequest::new(
                            session.clone(),
                            tool_call,
                            vec![
                                PermissionOption::new(
                                    "yes",
                                    "Apply this edit",
                                    PermissionOptionKind::AllowOnce,
                                ),
                                PermissionOption::new(
                                    "no",
                                    "Skip it",
                                    PermissionOptionKind::RejectOnce,
                                ),
                            ],
                        ))
                        .block_task()
                        .await?;

                    say(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                        ContentBlock::Text(TextContent::new(format!(
                            "verdict: {:?}
",
                            decision.outcome
                        ))),
                    )))?;

                    responder.respond(PromptResponse::new(StopReason::EndTurn))
                })?;
                Ok(())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_to(Stdio::new())
        .await
}
