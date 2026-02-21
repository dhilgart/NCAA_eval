# Claude Code — Workspace Instructions

## Bash Rules (Mandatory — No Exceptions)

- **Never use `git -C <path> <cmd>`** — use plain `git <cmd>`. The working directory is always the repo root (`/home/dhilg/git/NCAA_eval`); `-C` is never needed and is not pre-approved.
- **Never chain bash commands with `&&`, `|`, or `;`** — always issue commands as separate tool calls.
- **Never use heredocs or `printf ... > file` for commit messages** — use the `Write` tool to create `/tmp/commit_msg.txt`, then `ncaa-git commit -F /tmp/commit_msg.txt`.
- **Always use full path for git commits**: `/home/dhilg/bin/ncaa-git commit` (activates pre-commit hooks). Plain `git commit` skips the hooks.
