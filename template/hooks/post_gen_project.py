"""Post-generation hook for the cookiecutter template.

Handles conditional removal of optional components and project initialization.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

USE_BMAD = "{{ cookiecutter.use_bmad }}" == "y"
LICENSE = "{{ cookiecutter.open_source_license }}"


def remove_path(path: str) -> None:
    """Remove a file or directory if it exists."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)


def run(args: list[str], description: str) -> None:
    """Run a subprocess command with user-friendly error reporting.

    Args:
        args: Command and arguments to run.
        description: Human-readable description for error messages.

    Raises:
        SystemExit: On non-zero return code, prints the error and exits.
    """
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n\u274c Post-generation hook failed during: {description}")
        print(f"   Command: {' '.join(args)}")
        if result.stdout:
            print(f"   stdout: {result.stdout.strip()}")
        if result.stderr:
            print(f"   stderr: {result.stderr.strip()}")
        print("\nYour project files have been generated but git initialization failed.")
        print("To complete setup manually, run:")
        print("  git init -b main")
        print("  git config user.name '<your name>'")
        print("  git config user.email '<your email>'")
        print("  git add .")
        print("  git commit -m 'feat: initialize project from bmad-python-template'")
        sys.exit(1)


def main() -> None:
    """Run post-generation tasks."""
    # Remove BMAD directories if not using BMAD
    if not USE_BMAD:
        remove_path("_bmad")
        remove_path("_bmad-output")

    # Remove LICENSE file if no license selected
    if LICENSE == "None":
        remove_path("LICENSE")

    # Initialize git repository
    run(["git", "init", "-b", "main"], "git init")
    run(
        ["git", "config", "user.name", "{{ cookiecutter.author_name }}"],
        "git config user.name",
    )
    run(
        ["git", "config", "user.email", "{{ cookiecutter.author_email }}"],
        "git config user.email",
    )
    run(["git", "add", "."], "git add")
    run(
        ["git", "commit", "-m", "feat: initialize project from bmad-python-template"],
        "git commit",
    )

    print("")
    print("=" * 60)
    print("  Project {{ cookiecutter.project_slug }} created successfully!")
    print("=" * 60)
    print("")
    print("Next steps:")
    print("  cd {{ cookiecutter.project_slug }}")
    print("  conda create -n {{ cookiecutter.project_slug }} python={{ cookiecutter.python_version_min }} -y")
    print("  conda activate {{ cookiecutter.project_slug }}")
    print("  pip install poetry")
    print("  POETRY_VIRTUALENVS_CREATE=false poetry install --with dev")
    print("  pre-commit install")
    print("  pre-commit autoupdate  # Update hook versions to latest")
    print("  nox  # Run full quality pipeline")
    print("")


if __name__ == "__main__":
    main()
