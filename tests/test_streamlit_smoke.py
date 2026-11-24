"""Headless Streamlit AppTest smoke tests verifying all pages render without exceptions."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from web_app.config import WEB_APP_DIR


@pytest.mark.parametrize(
    "relative_script",
    [
        "app.py",
        "pages/1_search.py",
        "pages/2_report.py",
        "pages/3_clusters.py",
    ],
)
def test_streamlit_page_renders_without_exception(relative_script: str) -> None:
    script_path = WEB_APP_DIR / relative_script
    at = AppTest.from_file(str(script_path), default_timeout=30)
    at.run()
    assert not at.exception, f"Streamlit exception on {relative_script}: {at.exception}"


def test_search_page_interactive_compare_flow() -> None:
    script_path = WEB_APP_DIR / "pages/1_search.py"
    at = AppTest.from_file(str(script_path), default_timeout=30)
    at.run()
    at.button(key="btn_search_sample").click().run()
    assert not at.exception, f"Interactive search failed: {at.exception}"
    assert len(at.metric) >= 15
    assert len(at.dataframe) >= 5
