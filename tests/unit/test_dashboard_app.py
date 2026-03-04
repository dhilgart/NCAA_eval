"""Smoke tests for dashboard module imports and session-state initialisation."""

from __future__ import annotations

import importlib
from pathlib import Path


class TestDashboardImports:
    """Verify all dashboard modules are importable without error."""

    def test_import_filters(self) -> None:
        mod = importlib.import_module("dashboard.lib.filters")
        assert hasattr(mod, "get_data_dir")
        assert hasattr(mod, "load_available_years")
        assert hasattr(mod, "load_available_runs")
        assert hasattr(mod, "load_available_scorings")
        # Story 8.8: these were explicitly added to filters.__all__ as re-exports
        assert hasattr(mod, "load_data_freshness")
        assert hasattr(mod, "load_scoring_display_names")

    def test_import_styles(self) -> None:
        mod = importlib.import_module("dashboard.lib.styles")
        assert hasattr(mod, "MONOSPACE_CSS")

    def test_import_lib_package(self) -> None:
        importlib.import_module("dashboard.lib")

    def test_import_components_package(self) -> None:
        importlib.import_module("dashboard.components")

    def test_import_page_home(self) -> None:
        importlib.import_module("dashboard.pages.home")

    def test_import_page_lab(self) -> None:
        importlib.import_module("dashboard.pages.1_Lab")

    def test_import_page_presentation(self) -> None:
        importlib.import_module("dashboard.pages.2_Presentation")

    def test_import_page_model_deep_dive(self) -> None:
        importlib.import_module("dashboard.pages.3_Model_Deep_Dive")

    def test_import_page_pool_scorer(self) -> None:
        importlib.import_module("dashboard.pages.4_Pool_Scorer")


class TestFilterFunctionSignatures:
    """Verify filter functions are callable and decorated with cache_data."""

    def test_load_available_years_is_callable(self) -> None:
        from dashboard.lib.data_loaders import load_available_years

        assert callable(load_available_years)

    def test_load_available_runs_is_callable(self) -> None:
        from dashboard.lib.data_loaders import load_available_runs

        assert callable(load_available_runs)

    def test_load_available_scorings_is_callable(self) -> None:
        from dashboard.lib.data_loaders import load_available_scorings

        assert callable(load_available_scorings)

    def test_get_data_dir_is_callable(self) -> None:
        from dashboard.lib.data_loaders import get_data_dir

        assert callable(get_data_dir)

    def test_cached_functions_have_wrapped(self) -> None:
        """Streamlit's @cache_data stores the original via __wrapped__."""
        from dashboard.lib.data_loaders import (
            load_available_runs,
            load_available_scorings,
            load_available_years,
        )

        assert hasattr(load_available_years, "__wrapped__")
        assert hasattr(load_available_runs, "__wrapped__")
        assert hasattr(load_available_scorings, "__wrapped__")


class TestMonospaceCss:
    def test_css_contains_font_family(self) -> None:
        from dashboard.lib.styles import MONOSPACE_CSS

        assert "IBM Plex Mono" in MONOSPACE_CSS
        assert "monospace" in MONOSPACE_CSS

    def test_css_is_wrapped_in_style_tags(self) -> None:
        from dashboard.lib.styles import MONOSPACE_CSS

        assert "<style>" in MONOSPACE_CSS
        assert "</style>" in MONOSPACE_CSS


class TestLoadDataFreshness:
    """Unit tests for load_data_freshness."""

    def test_returns_dict_with_expected_keys(self, tmp_path: Path) -> None:
        from dashboard.lib.data_loaders import load_data_freshness

        result = load_data_freshness.__wrapped__(str(tmp_path))  # type: ignore[attr-defined]
        assert "last_sync_date" in result
        assert "latest_game_date" in result

    def test_nonexistent_dir_returns_none_values(self) -> None:
        from dashboard.lib.data_loaders import load_data_freshness

        result = load_data_freshness.__wrapped__("/nonexistent/path/that/does/not/exist")  # type: ignore[attr-defined]
        assert result["last_sync_date"] is None
        assert result["latest_game_date"] is None

    def test_empty_dir_returns_none_values(self, tmp_path: Path) -> None:
        from dashboard.lib.data_loaders import load_data_freshness

        result = load_data_freshness.__wrapped__(str(tmp_path))  # type: ignore[attr-defined]
        assert result["last_sync_date"] is None
        assert result["latest_game_date"] is None

    def test_parquet_file_sets_sync_date(self, tmp_path: Path) -> None:
        from dashboard.lib.data_loaders import load_data_freshness

        # Create a dummy parquet file under games/ (sync-data location, not runs/)
        parquet_file = tmp_path / "games" / "season=2025" / "data.parquet"
        parquet_file.parent.mkdir(parents=True)
        parquet_file.write_bytes(b"dummy")

        result = load_data_freshness.__wrapped__(str(tmp_path))  # type: ignore[attr-defined]
        # last_sync_date should be a date string (YYYY-MM-DD format)
        assert result["last_sync_date"] is not None
        assert len(result["last_sync_date"]) == 10
        assert result["last_sync_date"][4] == "-"

    def test_is_cached_with_wrapped(self) -> None:
        from dashboard.lib.data_loaders import load_data_freshness

        assert hasattr(load_data_freshness, "__wrapped__")


class TestSessionStateKeys:
    """Verify the expected session-state keys are referenced in app.py source."""

    def test_expected_keys_in_app_source(self) -> None:
        from pathlib import Path

        app_source = (Path(__file__).resolve().parent.parent.parent / "dashboard" / "app.py").read_text()
        for key in ("selected_year", "selected_run_id", "selected_scoring"):
            assert key in app_source, f"Session-state key {key!r} not found in app.py"
