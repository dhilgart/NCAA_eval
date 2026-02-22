"""Integration tests for the cookiecutter template.

Verifies that the template generates a valid project with correct
structure, parameterization, and tooling configuration.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

TEMPLATE_DIR = Path(__file__).parent.parent.parent / "template"

# Default context for test project generation
DEFAULT_CONTEXT = {
    "project_name": "Test Project",
    "project_slug": "test_project",
    "project_description": "A test project for template validation",
    "author_name": "Test Author",
    "author_email": "test@example.com",
    "github_username": "testuser",
    "python_version_min": "3.12",
    "open_source_license": "MIT",
    "use_bmad": "y",
    "bmad_user_name": "Test Author",
}


def _generate_project(
    tmp_path: Path,
    extra_context: dict[str, str] | None = None,
) -> Path:
    """Generate a project from the template using cookiecutter.

    Args:
        tmp_path: Temporary directory for generated output.
        extra_context: Override default context values.

    Returns:
        Path to the generated project root.
    """
    context = {**DEFAULT_CONTEXT, **(extra_context or {})}
    # Build cookiecutter CLI extra_context args
    cc_args = [f"{k}={v}" for k, v in context.items()]

    result = subprocess.run(
        [
            "cookiecutter",
            str(TEMPLATE_DIR),
            "--no-input",
            "--output-dir",
            str(tmp_path),
            *cc_args,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"cookiecutter failed:\n{result.stderr}"

    project_dir = tmp_path / context["project_slug"]
    assert project_dir.is_dir(), f"Generated project not found at {project_dir}"
    return project_dir


@pytest.mark.integration
@pytest.mark.slow
class TestTemplateGeneration:
    """Test that the template generates a valid project."""

    def test_generates_project_directory(self, tmp_path: Path) -> None:
        """Verify cookiecutter generates the project directory."""
        project = _generate_project(tmp_path)
        assert project.is_dir()
        assert project.name == "test_project"

    def test_src_layout_structure(self, tmp_path: Path) -> None:
        """Verify src layout with package directory."""
        project = _generate_project(tmp_path)
        src_pkg = project / "src" / "test_project"
        assert src_pkg.is_dir()
        assert (src_pkg / "__init__.py").is_file()
        assert (src_pkg / "py.typed").is_file()

    def test_tests_structure(self, tmp_path: Path) -> None:
        """Verify test directory structure."""
        project = _generate_project(tmp_path)
        tests_dir = project / "tests"
        assert tests_dir.is_dir()
        assert (tests_dir / "__init__.py").is_file()
        assert (tests_dir / "conftest.py").is_file()
        assert (tests_dir / "unit" / "__init__.py").is_file()
        assert (tests_dir / "unit" / "test_package.py").is_file()
        assert (tests_dir / "unit" / "test_example_property.py").is_file()
        assert (tests_dir / "integration" / "__init__.py").is_file()
        assert (tests_dir / "integration" / "test_example_integration.py").is_file()

    def test_docs_structure(self, tmp_path: Path) -> None:
        """Verify documentation files are generated."""
        project = _generate_project(tmp_path)
        docs_dir = project / "docs"
        assert (docs_dir / "conf.py").is_file()
        assert (docs_dir / "STYLE_GUIDE.md").is_file()
        assert (docs_dir / "TESTING_STRATEGY.md").is_file()
        assert (docs_dir / "BMAD_UPDATE_GUIDE.md").is_file()
        assert (docs_dir / "index.rst").is_file()

    def test_github_workflows(self, tmp_path: Path) -> None:
        """Verify GitHub Actions workflows are generated."""
        project = _generate_project(tmp_path)
        workflows = project / ".github" / "workflows"
        assert (workflows / "python-check.yaml").is_file()
        assert (workflows / "main-updated.yaml").is_file()

    def test_config_files_present(self, tmp_path: Path) -> None:
        """Verify essential config files are generated."""
        project = _generate_project(tmp_path)
        assert (project / "pyproject.toml").is_file()
        assert (project / "noxfile.py").is_file()
        assert (project / ".pre-commit-config.yaml").is_file()
        assert (project / ".gitignore").is_file()
        assert (project / ".editorconfig").is_file()
        assert (project / "CONTRIBUTING.md").is_file()
        assert (project / "README.md").is_file()
        assert (project / "CLAUDE.md").is_file()
        assert (project / "cookie-cutter-improvements.md").is_file()

    def test_pyproject_toml_parameterized(self, tmp_path: Path) -> None:
        """Verify pyproject.toml has correct parameterized values."""
        project = _generate_project(tmp_path)
        content = (project / "pyproject.toml").read_text()
        assert 'name = "test_project"' in content
        assert '"Test Author <test@example.com>"' in content
        assert 'python = ">=3.12,<4.0"' in content
        assert "strict = true" in content
        assert "line-length = 110" in content

    def test_noxfile_parameterized(self, tmp_path: Path) -> None:
        """Verify noxfile.py references the correct package name."""
        project = _generate_project(tmp_path)
        content = (project / "noxfile.py").read_text()
        assert "src/test_project" in content

    def test_readme_parameterized(self, tmp_path: Path) -> None:
        """Verify README.md has correct project name."""
        project = _generate_project(tmp_path)
        content = (project / "README.md").read_text()
        assert "# Test Project" in content
        assert "test_project" in content

    def test_init_py_parameterized(self, tmp_path: Path) -> None:
        """Verify __init__.py has correct docstring."""
        project = _generate_project(tmp_path)
        content = (project / "src" / "test_project" / "__init__.py").read_text()
        assert "Test Project" in content
        assert "from __future__ import annotations" in content


@pytest.mark.integration
@pytest.mark.slow
class TestTemplateBmadToggle:
    """Test BMAD integration toggle."""

    def test_bmad_included_when_enabled(self, tmp_path: Path) -> None:
        """Verify BMAD directories exist when use_bmad=y."""
        project = _generate_project(tmp_path, {"use_bmad": "y"})
        assert (project / "_bmad" / "bmm" / "config.yaml").is_file()
        assert (project / "_bmad-output").is_dir()

    def test_bmad_excluded_when_disabled(self, tmp_path: Path) -> None:
        """Verify BMAD directories removed when use_bmad=n."""
        project = _generate_project(tmp_path, {"use_bmad": "n"})
        assert not (project / "_bmad").exists()
        assert not (project / "_bmad-output").exists()


@pytest.mark.integration
@pytest.mark.slow
class TestTemplateLicenseToggle:
    """Test license selection."""

    def test_mit_license(self, tmp_path: Path) -> None:
        """Verify MIT license is generated."""
        project = _generate_project(tmp_path, {"open_source_license": "MIT"})
        content = (project / "LICENSE").read_text()
        assert "MIT License" in content
        assert "Test Author" in content

    def test_gpl_license(self, tmp_path: Path) -> None:
        """Verify GPL license is generated."""
        project = _generate_project(
            tmp_path,
            {"open_source_license": "GNU General Public License v3"},
        )
        content = (project / "LICENSE").read_text()
        assert "GNU GENERAL PUBLIC LICENSE" in content

    def test_no_license(self, tmp_path: Path) -> None:
        """Verify LICENSE removed when None selected."""
        project = _generate_project(tmp_path, {"open_source_license": "None"})
        assert not (project / "LICENSE").exists()


@pytest.mark.integration
@pytest.mark.slow
class TestTemplateCustomSlug:
    """Test custom project slug generation."""

    def test_custom_project_name_generates_slug(self, tmp_path: Path) -> None:
        """Verify project_slug is auto-derived from project_name."""
        project = _generate_project(
            tmp_path,
            {
                "project_name": "My Data Tool",
                "project_slug": "my_data_tool",
            },
        )
        assert project.name == "my_data_tool"
        assert (project / "src" / "my_data_tool" / "__init__.py").is_file()
        content = (project / "pyproject.toml").read_text()
        assert 'name = "my_data_tool"' in content
