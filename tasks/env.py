from __future__ import annotations

from invoke.context import Context
from invoke.tasks import task

from tasks.common import VENV_PREFIX


@task
def clean(ctx: Context) -> None:
    """Remove virtual environment"""
    ctx.run("pipenv --rm", warn=True)


@task
def init(ctx: Context) -> None:
    """Install production dependencies"""
    ctx.run("pipenv install --deploy")


@task
def setup_pre_commit_hook(ctx: Context) -> None:
    """Setup pre-commit hook to automate check before git commit and git push"""
    ctx.run("git init")
    for hook_type in ("pre-commit", "pre-push", "commit-msg"):
        ctx.run(f"{VENV_PREFIX} pre-commit install -t {hook_type}")


@task(optional=["no-pre-commit"])
def init_dev(ctx: Context, no_pre_commit: bool = False) -> None:
    """Install development dependencies and setup pre-commit hooks"""
    ctx.run("pipenv install --dev")
    if not no_pre_commit:
        setup_pre_commit_hook(ctx)
