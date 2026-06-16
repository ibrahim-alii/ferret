import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: hits real external APIs")
    config.addinivalue_line("markers", "eval: full RAG eval suite")
