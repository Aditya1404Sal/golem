// Copyright 2024-2026 Golem Cloud
//
// Licensed under the Golem Source License v1.1 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://license.golem.cloud/LICENSE
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! `golem agent shell` — drive an agent that presents a *shell surface* as an interactive REPL.
//!
//! The sibling of `agent stream` rather than a mode of it: `stream` tails the agent's output channels
//! one-way, whereas this invokes the agent and renders what comes back.
//!
//! Nothing here is specific to any one agent: the surface is validated structurally and the line is
//! shipped verbatim. The prompt reads `clank [/cwd] ❯` — the agent-derived label
//! ([`shell_prompt_label`] — `ClankAgent` shows `clank`, `RandomAgent` shows `random`), the working
//! directory in blue brackets when the agent reports one, and a starship-style `❯` marker right
//! before the input that is green after a success / red after a failure ([`prompt_message`]).
//!
//! **The contract.** `agent shell` requires a **durable** agent type exposing three methods, named
//! exactly as written (snake_case is normative — the surface is matched against the reflected method
//! names verbatim):
//!
//! | Method | Signature | Role |
//! |---|---|---|
//! | `eval` | `(string) -> eval-result` | run one command line, return its output |
//! | `answer_prompt` | `(string) -> eval-result` | deliver a human answer to an outstanding question |
//! | `abort_prompt` | `() -> eval-result` | cancel an outstanding question |
//!
//! where `eval-result` is the record `{ stdout: string, stderr: string, exit-code: u8,
//! pending-prompt: option<{ question: string, choices: option<list<string>> }> }`. An agent MAY
//! append a fifth `cwd: string` field; when present the shell shows it in blue brackets before the
//! `❯` marker so a `cd` is reflected (`clank ❯` → `clank [/app] ❯`). The record is decoded
//! positionally and the extra field is read optionally, so agents that omit it are entirely unaffected.
//!
//! An agent that does not present that surface gets a clear error, and **every agent is otherwise
//! unaffected** — `stream` log-streams exactly as before. Ephemeral agent types are rejected up
//! front: each invocation runs on a fresh instance, so session state could not persist between
//! lines and a pending question could never be answered.
//!
//! **Why the pause is a separate call.** Such an agent never blocks waiting on a human: a question is
//! recorded as durable state and returned *immediately* in `pending-prompt`; the answer arrives on a
//! **separate invocation**. That shape is forced by the runtime — agent invocations are serialized per
//! instance, so an agent parked on a human would be unreachable. It also means a prompt left dangling
//! by a previous session is picked up and resolved by the next one, which this loop does.

use anyhow::{anyhow, bail};
use golem_client::api::AgentClient;
use golem_client::model::{AgentInvocationMode, AgentInvocationRequest};
use golem_common::model::IdempotencyKey;
use golem_common::model::agent::AgentMode;
use golem_common::model::agent::ParsedAgentId;
use golem_common::schema::agent::{AgentMethodSchema, AgentTypeSchema, InputSchema};
use golem_common::schema::{FromSchema, SchemaType, SchemaValue};
use inquire::ui::{Color, RenderConfig, Styled};
use inquire::{InquireError, Select, Text};

use super::WorkerCommandHandler;
use crate::error::service::MapServiceError;
use crate::log::{LogColorize, log_action, log_error, log_preformatted, logln};
use crate::model::text::worker::format_agent_name_match;
use crate::model::worker::AgentNameMatch;

/// Run one command line and return its output.
const EVAL: &str = "eval";
/// Deliver a human answer to an outstanding question raised by [`EVAL`].
const ANSWER_PROMPT: &str = "answer_prompt";
/// Cancel an outstanding question raised by [`EVAL`].
const ABORT_PROMPT: &str = "abort_prompt";
/// Typed in at the answer prompt to cancel the outstanding question instead of answering it.
const ABORT_ANSWER: &str = ":abort";

/// The record every interactive-shell method returns.
#[derive(Debug, FromSchema)]
struct EvalResult {
    stdout: String,
    stderr: String,
    exit_code: u8,
    /// `Some` when the command paused on a question the caller must answer (via [`ANSWER_PROMPT`])
    /// before any further [`EVAL`] will run.
    pending_prompt: Option<PendingPrompt>,
}

/// A question the agent is waiting on. Field order is load-bearing — see [`EvalResult`].
#[derive(Debug, FromSchema)]
struct PendingPrompt {
    question: String,
    /// When present, the answer must be one of these.
    choices: Option<Vec<String>>,
}

/// Pull the optional working directory an agent MAY report as a **fifth** eval-result field.
///
/// The surface contract is `{ stdout, stderr, exit-code, pending-prompt }`; an agent may append a
/// `cwd: string` so the shell can show where it is and reflect a `cd` (clank does). The record is
/// decoded positionally and [`EvalResult`]'s derive ignores trailing fields, so this reads index 4
/// directly and yields `None` for any agent that omits it (or reports an empty path). Keeping it
/// optional is what lets `agent shell` stay generic across agents that predate the field.
fn extract_cwd(value: &SchemaValue) -> Option<String> {
    match value {
        SchemaValue::Record { fields } => match fields.get(4) {
            Some(SchemaValue::String(cwd)) if !cwd.is_empty() => Some(cwd.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Whether `agent_type` presents the interactive-shell surface, and why not if it doesn't.
pub fn validate_interactive_surface(agent_type: &AgentTypeSchema) -> anyhow::Result<()> {
    if agent_type.mode == AgentMode::Ephemeral {
        bail!(
            "Agent type {} cannot be used with `agent shell`: it is ephemeral, so each invocation \
             runs on a fresh instance — shell state would reset between lines and a pending \
             question could never be answered.",
            agent_type.type_name.0.log_color_highlight(),
        );
    }

    let find = |name: &str| agent_type.methods.iter().find(|m| m.name == name);

    let missing: Vec<&str> = [EVAL, ANSWER_PROMPT, ABORT_PROMPT]
        .into_iter()
        .filter(|m| find(m).is_none())
        .collect();

    if !missing.is_empty() {
        bail!(
            "Agent type {} cannot be used with `agent shell`: it has no {} method(s).\n\
             `agent shell` requires an agent exposing `{EVAL}(string)`, `{ANSWER_PROMPT}(string)` and \
             `{ABORT_PROMPT}()` (snake_case, matched verbatim), each returning an eval-result record \
             {{stdout, stderr, exit-code, pending-prompt}}.",
            agent_type.type_name.0.log_color_highlight(),
            missing.join(", ").log_color_error_highlight(),
        );
    }

    single_string_parameter(agent_type, find(EVAL).unwrap())?;
    single_string_parameter(agent_type, find(ANSWER_PROMPT).unwrap())?;
    zero_parameters(agent_type, find(ABORT_PROMPT).unwrap())?;

    Ok(())
}

/// Assert `method` takes exactly one `string` parameter, naming the actual mismatch (count vs type).
fn single_string_parameter(
    agent_type: &AgentTypeSchema,
    method: &AgentMethodSchema,
) -> anyhow::Result<()> {
    let InputSchema::Parameters(params) = &method.input_schema;
    match params.as_slice() {
        [only] if matches!(only.schema, SchemaType::String { .. }) => Ok(()),
        [only] => bail!(
            "Agent type {} cannot be used with `agent shell`: `{}` must take one string parameter, \
             but its parameter `{}` is not a string.",
            agent_type.type_name.0.log_color_highlight(),
            method.name,
            only.name,
        ),
        _ => bail!(
            "Agent type {} cannot be used with `agent shell`: `{}` must take exactly one string \
             parameter, but it takes {} parameters.",
            agent_type.type_name.0.log_color_highlight(),
            method.name,
            params.len(),
        ),
    }
}

/// Assert `method` takes no parameters (the CLI invokes it with an empty parameter record).
fn zero_parameters(agent_type: &AgentTypeSchema, method: &AgentMethodSchema) -> anyhow::Result<()> {
    let InputSchema::Parameters(params) = &method.input_schema;
    if params.is_empty() {
        Ok(())
    } else {
        bail!(
            "Agent type {} cannot be used with `agent shell`: `{}` must take no parameters, but it \
             takes {}.",
            agent_type.type_name.0.log_color_highlight(),
            method.name,
            params.len(),
        )
    }
}

/// Render an [`EvalResult`]'s output as it should appear in the terminal.
fn render(result: &EvalResult) -> String {
    let mut out = String::new();
    out.push_str(&result.stdout);
    if !result.stdout.is_empty() && !result.stdout.ends_with('\n') {
        out.push('\n');
    }
    if !result.stderr.is_empty() {
        out.push_str(&result.stderr);
        if !result.stderr.ends_with('\n') {
            out.push('\n');
        }
    }
    if result.exit_code != 0 {
        out.push_str(&format!("exit {}\n", result.exit_code));
    }
    out
}

/// The shell's prompt label, derived from the agent type so the surface reads as the agent's own:
/// kebab-case the type name and drop a trailing `-agent` (the conventional suffix carries no
/// information at a shell prompt). `ClankAgent` → `clank`, `GreeterAgent` → `greeter`, `RpcCounter`
/// → `rpc-counter`. A type named literally `Agent` (or an empty name) falls back to `agent`. The
/// prompt is terminated by the `❯` indicator ([`shell_render_config`]), so the label carries no `$`.
fn shell_prompt_label(type_name: &str) -> String {
    let kebab = heck::ToKebabCase::to_kebab_case(type_name);
    let stem = kebab.strip_suffix("-agent").unwrap_or(&kebab);
    if stem.is_empty() {
        "agent".to_string()
    } else {
        stem.to_string()
    }
}

/// ANSI reset — closes a colour span opened in a prompt segment (see [`prompt_message`]).
const ANSI_RESET: &str = "\x1b[0m";

/// The agent [label](shell_prompt_label) as the prompt's leading segment — bold bright-cyan when
/// `color`, plain otherwise. It goes in the render config's prefix so inquire places it first and
/// adds the separating space; the embedded ANSI renders verbatim (inquire excludes escapes from its
/// width math). `color` gates the escapes so the label is plain under NO_COLOR / a non-tty.
fn label_prefix(label: &str, color: bool) -> String {
    if color {
        format!("\x1b[1;96m{label}{ANSI_RESET}")
    } else {
        label.to_string()
    }
}

/// The prompt message that renders right before the input: the working directory in blue brackets
/// (when the agent reports one) then the `❯` marker — bright-green after a success, bright-red after
/// a failure. It is baked into the MESSAGE, not the render config, because inquire has no marker
/// slot between the message and the input; it renders the embedded ANSI without disturbing cursor
/// alignment (escapes are excluded from its width math). A `None`/empty cwd yields just the marker
/// (→ `clank ❯`). `color` gates every escape so the string is plain under NO_COLOR / a non-tty.
fn prompt_message(cwd: Option<&str>, ok: bool, color: bool) -> String {
    let marker = match (color, ok) {
        (false, _) => "❯".to_string(),
        (true, true) => format!("\x1b[92m❯{ANSI_RESET}"),
        (true, false) => format!("\x1b[91m❯{ANSI_RESET}"),
    };
    match cwd {
        Some(cwd) if !cwd.is_empty() => {
            let cwd = if color {
                format!("\x1b[34m[{cwd}]{ANSI_RESET}")
            } else {
                format!("[{cwd}]")
            };
            format!("{cwd} {marker}")
        }
        _ => marker,
    }
}

/// inquire styling for the command prompt: the agent label ([`label_prefix`]) leads as the prefix —
/// inquire appends the separating space and renders the label's own ANSI verbatim. The cwd and `❯`
/// marker come from the message ([`prompt_message`]) so the marker sits right before the input, and
/// `render_config.prompt` is left at its empty default so inquire does not re-wrap that ANSI. The
/// answered prefix reuses the label so submitted (history) lines read identically.
fn shell_render_config(prefix: &str) -> RenderConfig<'_> {
    RenderConfig::default()
        .with_prompt_prefix(Styled::new(prefix))
        .with_answered_prompt_prefix(Styled::new(prefix))
}

/// inquire styling for a human-answer sub-prompt (the `answer:` line / choice picker a pending
/// question raises): the same `❯` indicator in a neutral colour, visually distinct from the
/// command line so it is clear the shell is waiting on an answer, not a command.
fn answer_render_config() -> RenderConfig<'static> {
    RenderConfig::default().with_prompt_prefix(Styled::new("❯").with_fg(Color::DarkYellow))
}

/// Format a pending question for display, appending its allowed answers when it has them.
fn format_question(prompt: &PendingPrompt) -> String {
    match &prompt.choices {
        Some(choices) if !choices.is_empty() => {
            format!("{} [{}]", prompt.question, choices.join(", "))
        }
        _ => prompt.question.clone(),
    }
}

/// Discard any keystrokes the terminal buffered while an invocation was in flight, so they are not
/// replayed into the next inquire prompt. Between prompts the tty sits in cooked mode and queues
/// everything typed; inquire has no "clear pending input" option, so the drain happens here, with
/// crossterm's zero-timeout poll under a raw-mode guard (the same enable/disable pairing the REPL
/// supervisor uses). Best-effort: any terminal error just skips the drain — worse output beats a
/// broken shell.
fn drain_typeahead() {
    if crossterm::terminal::enable_raw_mode().is_err() {
        return;
    }
    drain_events(
        || matches!(crossterm::event::poll(std::time::Duration::ZERO), Ok(true)),
        || crossterm::event::read().is_ok(),
    );
    let _ = crossterm::terminal::disable_raw_mode();
}

/// The drain loop over injected probes, separated for testability: `pending` reports whether an
/// event is queued, `consume` reads one (returning false on failure, which stops the drain).
/// Returns the number of events discarded.
fn drain_events(mut pending: impl FnMut() -> bool, mut consume: impl FnMut() -> bool) -> usize {
    let mut discarded = 0;
    while pending() {
        if !consume() {
            break;
        }
        discarded += 1;
    }
    discarded
}

/// The client terminal's width in columns, or 80 when it can't be determined. Sent to the agent as
/// `export COLUMNS=<w>` so a terminal-style `ls` (and other columnar output) fills the real window —
/// the agent has no terminal of its own to measure.
fn term_width() -> u16 {
    crossterm::terminal::size().map_or(80, |(cols, _)| cols)
}

/// Await `fut` while animating a `clank`-themed loader in place, so the one-time cold start of a
/// fresh agent instance (loading + instantiating the wasm, building the shell) reads as "warming up"
/// rather than a hang. No animation when stderr isn't a terminal (piped/scripted) — the future is
/// just awaited. The loader line is cleared before returning, so the prompt draws cleanly after.
async fn with_clank_loader<F: std::future::Future>(fut: F) -> F::Output {
    use std::io::{IsTerminal, Write};
    if !std::io::stderr().is_terminal() {
        return fut.await;
    }
    draw_clank_loader(0);
    tokio::pin!(fut);
    let mut tick = 1u64;
    let out = loop {
        tokio::select! {
            biased;
            r = &mut fut => break r,
            () = tokio::time::sleep(std::time::Duration::from_millis(90)) => {
                draw_clank_loader(tick);
                tick += 1;
            }
        }
    };
    // Carriage return + erase-to-end-of-line, so the prompt starts on a clean line.
    eprint!("\r\x1b[2K");
    let _ = std::io::stderr().flush();
    out
}

/// One frame of the loader: a braille spinner, a cycling `clank`-themed status, and an indeterminate
/// pulse bar whose lit window bounces back and forth. Redrawn in place with a carriage return.
fn draw_clank_loader(tick: u64) {
    use std::io::Write;
    const SPIN: [&str; 8] = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];
    const MSG: [&str; 7] = [
        "initializing clank",
        "clanking the parts together",
        "greasing the gears",
        "warming up the shell",
        "tightening the bolts",
        "spinning up the agent",
        "almost there",
    ];
    let t = tick as usize;
    let spin = SPIN[t % SPIN.len()];
    let msg = MSG[(t / 16) % MSG.len()]; // ~1.5s per message at 90ms/frame
    // Pulse bar: a 3-cell lit window bounces across `width` cells (indeterminate progress).
    let (width, window) = (14usize, 3usize);
    let span = (width - window) * 2;
    let phase = t % span;
    let pos = if phase <= width - window {
        phase
    } else {
        span - phase
    };
    let bar: String = (0..width)
        .map(|i| {
            if i >= pos && i < pos + window {
                '█'
            } else {
                '·'
            }
        })
        .collect();
    if colored::control::SHOULD_COLORIZE.should_colorize() {
        eprint!("\r\x1b[96m{spin}\x1b[0m \x1b[1m{msg}\x1b[0m \x1b[90m▕{bar}▏\x1b[0m\x1b[K");
    } else {
        eprint!("\r{spin} {msg} [{bar}]\x1b[K");
    }
    let _ = std::io::stderr().flush();
}

/// Await a command `eval` while showing a minimal `⣾ {thinking|working}… {n}s` ticker, so a slow
/// invocation — above all a multi-second `ask` (which runs as one atomic eval, returning nothing
/// until it finishes) — reads as working, not frozen. Only on a terminal, and only after ~350ms so
/// quick commands never flash it. Label is `thinking` for an `ask`, else `working`. Cleared before
/// the result renders. (The connect warm-up keeps its own richer [`with_clank_loader`].)
async fn with_thinking_ticker<F: std::future::Future>(fut: F, is_ask: bool) -> F::Output {
    use std::io::{IsTerminal, Write};
    if !std::io::stderr().is_terminal() {
        return fut.await;
    }
    let label = if is_ask { "thinking" } else { "working" };
    const SPIN: [&str; 8] = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];
    let start = std::time::Instant::now();
    tokio::pin!(fut);
    let (mut frame, mut shown) = (0usize, false);
    let out = loop {
        tokio::select! {
            biased;
            r = &mut fut => break r,
            () = tokio::time::sleep(std::time::Duration::from_millis(110)) => {
                let elapsed = start.elapsed();
                if elapsed < std::time::Duration::from_millis(350) {
                    continue;
                }
                shown = true;
                let spin = SPIN[frame % SPIN.len()];
                frame += 1;
                eprint!("\r\x1b[96m{spin}\x1b[0m {label}\u{2026} {}s\x1b[K", elapsed.as_secs());
                let _ = std::io::stderr().flush();
            }
        }
    };
    if shown {
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }
    out
}

impl WorkerCommandHandler {
    /// Drive the interactive shell: read a line, invoke `eval`, render, resolve any pause, repeat.
    /// Runs until EOF (Ctrl-D), Ctrl-C, or `exit`.
    pub(super) async fn run_interactive_shell(
        &self,
        agent_name_match: &AgentNameMatch,
        agent_type: &AgentTypeSchema,
        agent_id: &ParsedAgentId,
    ) -> anyhow::Result<()> {
        log_action(
            "Connecting",
            format!(
                "to agent {} (Ctrl-D, Esc, or `exit` to leave)",
                format_agent_name_match(agent_name_match)
            ),
        );
        logln("");

        let label = shell_prompt_label(&agent_type.type_name.0);
        // Colour gating: reuse `colored`'s decision (NO_COLOR / CLICOLOR / tty) so the prompt matches
        // the rest of the CLI's output. The label prefix is fixed for the session; the cwd + `❯`
        // marker are rebuilt each line.
        let color = colored::control::SHOULD_COLORIZE.should_colorize();
        let prefix = label_prefix(&label, color);
        // Drives the `❯` marker colour: green after a success, red after a failure.
        let mut last_ok = true;
        // Sync the agent's COLUMNS to the client terminal width so a terminal-style `ls` fills the
        // window (the agent has no terminal of its own to measure). This also seeds the cwd: `export`
        // is a real eval, so its result still carries the agent's cwd — so the FIRST prompt shows it,
        // exactly as the old `eval("")` seed did. Best-effort and silent (result discarded); any
        // dangling prompt it re-surfaces is picked up by the first real command, and a no-cwd agent
        // just gets the bare label.
        let mut last_cols = term_width();
        // The seed is the first invocation on a fresh instance, so it pays the agent's cold start
        // (~seconds). Animate a `clank` loader over it so the wait reads as work, not a hang.
        let mut cwd: Option<String> = match with_clank_loader(self.interactive_invoke(
            agent_name_match,
            agent_type,
            agent_id,
            EVAL,
            Some(format!("export COLUMNS={last_cols}")),
        ))
        .await
        {
            Ok((_, seeded)) => seeded,
            Err(_) => None,
        };
        loop {
            let message = prompt_message(cwd.as_deref(), last_ok, color);
            let line = match Text::new(&message)
                .with_render_config(shell_render_config(&prefix))
                .prompt()
            {
                Ok(line) => line,
                // Ctrl-D / Esc / Ctrl-C are ordinary ways to leave a shell, not errors.
                Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => break,
                Err(err) => return Err(err.into()),
            };

            let line = line.trim();
            if line == "exit" {
                break;
            }
            if line.is_empty() {
                continue;
            }

            // Keep the agent's COLUMNS in sync when the client window was resized between commands,
            // so the next `ls` reflows. Silent and best-effort; only re-sent when the width changed.
            let cols = term_width();
            if cols != last_cols {
                last_cols = cols;
                let _ = self
                    .interactive_invoke(
                        agent_name_match,
                        agent_type,
                        agent_id,
                        EVAL,
                        Some(format!("export COLUMNS={cols}")),
                    )
                    .await;
            }

            let is_ask = line.split_whitespace().next() == Some("ask");
            let (result, invoke_cwd) = match with_thinking_ticker(
                self.interactive_invoke(
                    agent_name_match,
                    agent_type,
                    agent_id,
                    EVAL,
                    Some(line.to_string()),
                ),
                is_ask,
            )
            .await
            {
                Ok(pair) => pair,
                Err(err) => {
                    log_error(format!("{err:#}"));
                    last_ok = false;
                    continue;
                }
            };

            let (exit_code, final_cwd) = self
                .render_and_resolve(agent_name_match, agent_type, agent_id, result, invoke_cwd)
                .await;
            last_ok = exit_code == 0;
            // Keep the last reported cwd if this turn didn't carry one (e.g. an errored answer).
            if final_cwd.is_some() {
                cwd = final_cwd;
            }
            // A blank line after each command's output so the next prompt has room to breathe and
            // doesn't sit flush against the result — a consistent, terminal-like rhythm.
            logln("");
        }

        Ok(())
    }

    /// Invoke one method with an optional single string argument, decode its [`EvalResult`], and
    /// pull the optional working directory it may report ([`extract_cwd`]) so the caller can track
    /// it in the prompt.
    async fn interactive_invoke(
        &self,
        agent_name_match: &AgentNameMatch,
        agent_type: &AgentTypeSchema,
        agent_id: &ParsedAgentId,
        method: &str,
        argument: Option<String>,
    ) -> anyhow::Result<(EvalResult, Option<String>)> {
        let method_parameters = SchemaValue::Record {
            fields: argument.into_iter().map(SchemaValue::String).collect(),
        };

        let environment = &agent_name_match.environment;
        let idempotency_key = IdempotencyKey::fresh();
        let request = AgentInvocationRequest {
            app_name: environment.application_name.to_string(),
            env_name: environment.environment_name.to_string(),
            agent_type_name: agent_id.agent_type.0.clone(),
            parameters: agent_id.parameters.value().clone(),
            phantom_id: agent_id.phantom_id,
            config: None,
            method_name: method.to_string(),
            method_parameters,
            mode: AgentInvocationMode::Await,
            schedule_at: None,
            idempotency_key: Some(idempotency_key.value.clone()),
            deployment_revision: None,
            owner_account_email: None,
        };

        let clients = self.ctx.golem_clients().await?;
        let result = clients
            .agent
            .invoke_agent(Some(&idempotency_key.value), &request)
            .await;
        // While the invocation was in flight there was no prompt on screen, but the terminal (in
        // cooked mode between inquire prompts) kept buffering keystrokes — without a drain they
        // replay into the NEXT prompt, so commands typed during a long `eval` (a ~30s multi-turn
        // `ask`) land against later lines and the session looks reordered. Discard them on every
        // path, success or error, before anything re-prompts.
        drain_typeahead();
        let result = result.map_service_error()?;

        let typed = result.result.ok_or_else(|| {
            anyhow!(
                "Agent type {} returned no value from `{method}`; `agent shell` requires it to return \
                 an eval-result record.",
                agent_type.type_name.0
            )
        })?;

        let value = typed.value();
        let decoded = EvalResult::from_value(value).map_err(|err| {
            anyhow!(
                "Could not decode the eval-result returned by `{method}` on agent type {}: {err}",
                agent_type.type_name.0
            )
        })?;
        // Optional trailing `cwd` (5th field): agents that predate it return four fields → `None`.
        let cwd = extract_cwd(value);
        Ok((decoded, cwd))
    }

    /// Print a result and, while it carries a pending question, ask the human and deliver the answer.
    /// Returns the final exit code (so the caller can colour the next prompt's indicator) and the
    /// last working directory the agent reported (so the caller can track it in the prompt); a
    /// failure to invoke or to read an answer returns a non-zero code and the cwd known so far.
    /// `cwd` threads in the directory from the invocation that produced `result`.
    async fn render_and_resolve(
        &self,
        agent_name_match: &AgentNameMatch,
        agent_type: &AgentTypeSchema,
        agent_id: &ParsedAgentId,
        mut result: EvalResult,
        mut cwd: Option<String>,
    ) -> (u8, Option<String>) {
        loop {
            let output = render(&result);
            if !output.is_empty() {
                log_preformatted(&output);
            }

            let Some(prompt) = result.pending_prompt.take() else {
                return (result.exit_code, cwd);
            };

            let (method, argument) = match self.ask_human(&prompt) {
                Ok(Some(answer)) => (ANSWER_PROMPT, Some(answer)),
                Ok(None) => (ABORT_PROMPT, None),
                Err(err) => {
                    log_error(format!("{err:#}"));
                    return (1, cwd);
                }
            };

            let (next, next_cwd) = match self
                .interactive_invoke(agent_name_match, agent_type, agent_id, method, argument)
                .await
            {
                Ok(pair) => pair,
                Err(err) => {
                    log_error(format!("{err:#}"));
                    return (1, cwd);
                }
            };
            result = next;
            // Resolving a prompt can also change directory; keep the freshest reported cwd.
            if next_cwd.is_some() {
                cwd = next_cwd;
            }
        }
    }

    /// Ask the human the agent's question. `Ok(None)` means "abort this question".
    fn ask_human(&self, prompt: &PendingPrompt) -> anyhow::Result<Option<String>> {
        logln(format!("⏸  {}", format_question(prompt)));

        let answer = match &prompt.choices {
            Some(choices) if !choices.is_empty() => {
                let mut options = choices.clone();
                options.push(ABORT_ANSWER.to_string());
                Select::new("answer:", options)
                    .with_render_config(answer_render_config())
                    .prompt()
            }
            _ => Text::new(&format!("answer (`{ABORT_ANSWER}` to cancel):"))
                .with_render_config(answer_render_config())
                .prompt(),
        };

        match answer {
            Ok(answer) if answer.trim() == ABORT_ANSWER => Ok(None),
            Ok(answer) => Ok(Some(answer)),
            // Ctrl-D / Esc / Ctrl-C at a question aborts the question, not the shell.
            Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use golem_common::schema::SchemaValue;
    use pretty_assertions::assert_eq;
    use test_r::test;

    /// Build the schema-value an agent would return for an `eval-result`, so the tests exercise the
    /// real decode path rather than a hand-written guess at the wire JSON.
    fn eval_result_value(
        stdout: &str,
        stderr: &str,
        exit_code: u8,
        pending: Option<SchemaValue>,
    ) -> SchemaValue {
        SchemaValue::Record {
            fields: vec![
                SchemaValue::String(stdout.to_string()),
                SchemaValue::String(stderr.to_string()),
                SchemaValue::U8(exit_code),
                SchemaValue::Option {
                    inner: pending.map(Box::new),
                },
            ],
        }
    }

    fn pending_prompt_value(question: &str, choices: Option<&[&str]>) -> SchemaValue {
        SchemaValue::Record {
            fields: vec![
                SchemaValue::String(question.to_string()),
                SchemaValue::Option {
                    inner: choices.map(|choices| {
                        Box::new(SchemaValue::List {
                            elements: choices
                                .iter()
                                .map(|c| SchemaValue::String(c.to_string()))
                                .collect(),
                        })
                    }),
                },
            ],
        }
    }

    #[test]
    fn decodes_a_plain_result() {
        let value = eval_result_value("hi\n", "", 0, None);
        let result = EvalResult::from_value(&value).expect("decode");
        assert_eq!(result.stdout, "hi\n");
        assert_eq!(result.stderr, "");
        assert_eq!(result.exit_code, 0);
        assert!(result.pending_prompt.is_none());
    }

    #[test]
    fn decodes_a_pending_prompt_with_choices() {
        let value = eval_result_value(
            "pick one\n",
            "",
            0,
            Some(pending_prompt_value("pick one", Some(&["alpha", "beta"]))),
        );
        let result = EvalResult::from_value(&value).expect("decode");
        let prompt = result.pending_prompt.expect("pending prompt");
        assert_eq!(prompt.question, "pick one");
        assert_eq!(
            prompt.choices.as_deref(),
            Some(["alpha".to_string(), "beta".to_string()].as_slice())
        );
    }

    #[test]
    fn decodes_a_pending_prompt_without_choices() {
        let value = eval_result_value("q?\n", "", 0, Some(pending_prompt_value("q?", None)));
        let result = EvalResult::from_value(&value).expect("decode");
        let prompt = result.pending_prompt.expect("pending prompt");
        assert_eq!(prompt.question, "q?");
        assert!(prompt.choices.is_none());
    }

    #[test]
    fn a_mismatched_shape_is_an_error() {
        let value = SchemaValue::Record {
            fields: vec![SchemaValue::String("only one field".to_string())],
        };
        assert!(EvalResult::from_value(&value).is_err());
    }

    /// Append a fifth `cwd` field to a 4-field eval-result record (what a cwd-reporting agent sends).
    fn with_cwd(base: SchemaValue, cwd: &str) -> SchemaValue {
        let SchemaValue::Record { mut fields } = base else {
            unreachable!("eval_result_value builds a Record")
        };
        fields.push(SchemaValue::String(cwd.to_string()));
        SchemaValue::Record { fields }
    }

    #[test]
    fn extract_cwd_reads_the_fifth_field() {
        let value = with_cwd(eval_result_value("hi\n", "", 0, None), "/app");
        assert_eq!(extract_cwd(&value), Some("/app".to_string()));
    }

    #[test]
    fn extract_cwd_is_none_for_a_four_field_agent() {
        // The generic contract: an agent that omits cwd (four fields) yields None, never an error.
        let value = eval_result_value("hi\n", "", 0, None);
        assert_eq!(extract_cwd(&value), None);
    }

    #[test]
    fn extract_cwd_treats_empty_as_absent() {
        let value = with_cwd(eval_result_value("", "", 0, None), "");
        assert_eq!(extract_cwd(&value), None);
    }

    #[test]
    fn a_five_field_result_still_decodes_its_core_fields() {
        // The positional derive ignores the trailing cwd field, so the core EvalResult is unchanged.
        let value = with_cwd(eval_result_value("out\n", "err\n", 3, None), "/tmp");
        let result = EvalResult::from_value(&value).expect("decode ignores the trailing field");
        assert_eq!(result.stdout, "out\n");
        assert_eq!(result.stderr, "err\n");
        assert_eq!(result.exit_code, 3);
        assert_eq!(extract_cwd(&value), Some("/tmp".to_string()));
    }

    #[test]
    fn prompt_message_brackets_the_cwd_before_the_marker() {
        // Colour off → deterministic plain text: `[cwd]` then the `❯` marker, in that order.
        assert_eq!(prompt_message(Some("/app"), true, false), "[/app] ❯");
    }

    #[test]
    fn prompt_message_is_just_the_marker_without_a_cwd() {
        assert_eq!(prompt_message(None, true, false), "❯");
        assert_eq!(prompt_message(Some(""), true, false), "❯");
    }

    #[test]
    fn prompt_message_honours_no_colour() {
        // color=false emits zero escape bytes, with or without a cwd.
        assert!(!prompt_message(Some("/app"), true, false).contains('\x1b'));
        assert!(!prompt_message(None, false, false).contains('\x1b'));
    }

    #[test]
    fn label_prefix_colours_only_when_enabled() {
        assert_eq!(label_prefix("clank", false), "clank");
        assert!(label_prefix("clank", true).contains("\x1b[1;96mclank"));
    }

    #[test]
    fn render_emits_stdout_then_stderr_and_notes_a_failure() {
        let result = EvalResult {
            stdout: "out\n".to_string(),
            stderr: "err\n".to_string(),
            exit_code: 1,
            pending_prompt: None,
        };
        assert_eq!(render(&result), "out\nerr\nexit 1\n");
    }

    #[test]
    fn render_of_a_clean_result_is_just_its_stdout() {
        let result = EvalResult {
            stdout: "hi\n".to_string(),
            stderr: String::new(),
            exit_code: 0,
            pending_prompt: None,
        };
        assert_eq!(render(&result), "hi\n");
    }

    #[test]
    fn render_terminates_unterminated_output() {
        let result = EvalResult {
            stdout: "no newline".to_string(),
            stderr: String::new(),
            exit_code: 0,
            pending_prompt: None,
        };
        assert_eq!(render(&result), "no newline\n");
    }

    #[test]
    fn render_of_an_empty_success_is_empty() {
        let result = EvalResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
            pending_prompt: None,
        };
        assert_eq!(render(&result), "");
    }

    #[test]
    fn a_question_shows_its_choices() {
        let prompt = PendingPrompt {
            question: "Deploy?".to_string(),
            choices: Some(vec!["yes".to_string(), "no".to_string()]),
        };
        assert_eq!(format_question(&prompt), "Deploy? [yes, no]");
    }

    #[test]
    fn a_free_form_question_shows_no_brackets() {
        let prompt = PendingPrompt {
            question: "your name?".to_string(),
            choices: None,
        };
        assert_eq!(format_question(&prompt), "your name?");
    }

    #[test]
    fn prompt_label_drops_the_agent_suffix() {
        assert_eq!(shell_prompt_label("ClankAgent"), "clank");
        assert_eq!(shell_prompt_label("GreeterAgent"), "greeter");
    }

    #[test]
    fn prompt_label_kebab_cases_multiword_names() {
        assert_eq!(shell_prompt_label("RpcCounter"), "rpc-counter");
        assert_eq!(shell_prompt_label("RevisionEnvAgent"), "revision-env");
    }

    #[test]
    fn prompt_label_never_goes_empty() {
        // A type named literally `Agent` must not strip down to nothing.
        assert_eq!(shell_prompt_label("Agent"), "agent");
        assert_eq!(shell_prompt_label(""), "agent");
    }

    #[test]
    fn prompt_message_marker_colour_tracks_the_last_exit() {
        // With colour on, the `❯` marker is bright-green after a success and bright-red after a
        // failure; the cwd stays blue regardless.
        let ok = prompt_message(Some("/app"), true, true);
        let fail = prompt_message(Some("/app"), false, true);
        assert!(
            ok.contains("\x1b[92m❯"),
            "ok marker should be green: {ok:?}"
        );
        assert!(
            fail.contains("\x1b[91m❯"),
            "fail marker should be red: {fail:?}"
        );
        assert!(ok.contains("\x1b[34m[/app]"), "cwd should be blue: {ok:?}");
    }

    // --- validate_interactive_surface -------------------------------------------------------

    use golem_common::base_model::Empty;
    use golem_common::model::agent::{AgentTypeName, Snapshotting};
    use golem_common::schema::SchemaGraph;
    use golem_common::schema::agent::{AgentConstructorSchema, NamedField, OutputSchema};

    fn method(name: &str, params: Vec<NamedField>) -> AgentMethodSchema {
        AgentMethodSchema {
            name: name.to_string(),
            description: String::new(),
            prompt_hint: None,
            input_schema: InputSchema::Parameters(params),
            output_schema: OutputSchema::Unit,
            http_endpoint: Vec::new(),
            read_only: None,
        }
    }

    fn string_param(name: &str) -> NamedField {
        NamedField::user_supplied(
            name,
            SchemaType::String {
                metadata: Default::default(),
            },
        )
    }

    fn u8_param(name: &str) -> NamedField {
        NamedField::user_supplied(
            name,
            SchemaType::U8 {
                metadata: Default::default(),
                restrictions: Default::default(),
            },
        )
    }

    fn agent_type(mode: AgentMode, methods: Vec<AgentMethodSchema>) -> AgentTypeSchema {
        AgentTypeSchema {
            type_name: AgentTypeName("TestAgent".to_string()),
            description: String::new(),
            source_language: String::new(),
            schema: SchemaGraph::empty(),
            constructor: AgentConstructorSchema {
                name: None,
                description: String::new(),
                prompt_hint: None,
                input_schema: InputSchema::Parameters(Vec::new()),
            },
            methods,
            dependencies: Vec::new(),
            mode,
            http_mount: None,
            snapshotting: Snapshotting::Disabled(Empty {}),
            config: Vec::new(),
        }
    }

    fn shell_surface() -> Vec<AgentMethodSchema> {
        vec![
            method(EVAL, vec![string_param("cmd")]),
            method(ANSWER_PROMPT, vec![string_param("response")]),
            method(ABORT_PROMPT, Vec::new()),
        ]
    }

    #[test]
    fn a_conforming_durable_surface_validates() {
        let agent = agent_type(AgentMode::Durable, shell_surface());
        assert!(validate_interactive_surface(&agent).is_ok());
    }

    #[test]
    fn an_ephemeral_agent_is_rejected_up_front() {
        let agent = agent_type(AgentMode::Ephemeral, shell_surface());
        let err = validate_interactive_surface(&agent)
            .unwrap_err()
            .to_string();
        assert!(err.contains("ephemeral"), "err: {err}");
    }

    #[test]
    fn an_abort_prompt_with_parameters_is_rejected_at_connect() {
        let agent = agent_type(
            AgentMode::Durable,
            vec![
                method(EVAL, vec![string_param("cmd")]),
                method(ANSWER_PROMPT, vec![string_param("response")]),
                method(ABORT_PROMPT, vec![string_param("reason")]),
            ],
        );
        let err = validate_interactive_surface(&agent)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("abort_prompt") && err.contains("no parameters"),
            "err: {err}"
        );
    }

    #[test]
    fn a_wrong_typed_parameter_is_diagnosed_as_a_type_problem() {
        let agent = agent_type(
            AgentMode::Durable,
            vec![
                method(EVAL, vec![u8_param("cmd")]),
                method(ANSWER_PROMPT, vec![string_param("response")]),
                method(ABORT_PROMPT, Vec::new()),
            ],
        );
        let err = validate_interactive_surface(&agent)
            .unwrap_err()
            .to_string();
        assert!(err.contains("is not a string"), "err: {err}");
        assert!(!err.contains("takes 1"), "must not misreport arity: {err}");
    }

    #[test]
    fn a_missing_method_is_named_in_the_error() {
        let agent = agent_type(
            AgentMode::Durable,
            vec![method(EVAL, vec![string_param("cmd")])],
        );
        let err = validate_interactive_surface(&agent)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("answer_prompt") && err.contains("abort_prompt"),
            "err: {err}"
        );
    }

    #[test]
    fn drain_consumes_every_pending_event() {
        let queued = std::cell::Cell::new(3usize);
        let discarded = drain_events(
            || queued.get() > 0,
            || {
                queued.set(queued.get() - 1);
                true
            },
        );
        assert_eq!(discarded, 3);
        assert_eq!(queued.get(), 0);
    }

    #[test]
    fn drain_with_nothing_pending_is_a_noop() {
        assert_eq!(drain_events(|| false, || panic!("must not read")), 0);
    }

    #[test]
    fn drain_stops_on_a_failed_read() {
        // `pending` forever true, but the first read fails: the drain must bail, not spin.
        assert_eq!(drain_events(|| true, || false), 0);
    }
}
