# Prompt Overrides

This folder holds file-based prompt overrides for this workspace. Everything
here is read at request time — editing a file takes effect on the next turn,
with no rebuild and no restart.

## System prompt templates

These are the Jinja2 templates nanobot renders into the agent's system prompt.
A default copy is seeded here automatically when the workspace is created; edit
it and the next turn picks it up.

| File | What it controls |
|---|---|
| `agent/identity.md` | The opening identity / runtime / workspace section |
| `agent/platform_policy.md` | Platform-specific command and path rules |
| `agent/tool_contract.md` | Tool-calling contract and output conventions |
| `agent/skills_section.md` | How the skill catalogue is presented |
| `agent/subagent_system.md` | System prompt for spawned subagents |

Two rules worth knowing before you edit:

- **Template variables still apply.** These files are rendered with Jinja2, so
  `{{ workspace_path }}` / `{{ runtime }}` / `{% if channel == 'cli' %}` keep
  working in your copy. Delete a variable and that piece of context is simply
  gone — which is usually not what you want.
- **A broken override falls back to the built-in.** A syntax error is logged and
  nanobot renders the bundled template instead, so a typo here can't wedge your
  session. Emptying or deleting the file restores the default outright.

Only the files listed above can be overridden. Shared snippets under
`agent/_snippets/` are intentionally *not* overridable, so a security review
always has one place to read the real text.

## Dream memory

`dream.md` tells Dream how to organize memory in this workspace. Most users do not need to touch it. To create an editable copy, run:

```text
/dream-prompt init
```

That creates `prompts/dream.md`. Edit it in plain Markdown. Delete or empty it to return to nanobot's default memory behavior.

## Heartbeat evaluator

`evaluator.md` overrides the system prompt for the heartbeat notification gate — the model that decides whether a heartbeat result is worth delivering. This is an advanced override; you rarely need it. Before editing, read the evaluator code and the default `evaluator.md`.

To create an editable copy, run:

```text
/evaluator-prompt init
```

That creates `prompts/evaluator.md`. It must still instruct the model to call the `evaluate_notification` tool; otherwise the gate fails closed and stays silent. Delete or empty the file to return to the built-in prompt.
