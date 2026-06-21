import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: hits real external APIs")
    config.addinivalue_line("markers", "eval: full RAG eval suite")


@pytest.fixture(autouse=True)
def _disable_rate_limiter():
    """Turn off the slowapi limiter for every test.

    The limiter keeps an in-memory window keyed by client IP; under the test
    client all requests share one key, so counts would otherwise accumulate
    across unrelated tests and trip the limit non-deterministically.
    """
    from backend.api.limiter import limiter

    previous = limiter.enabled
    limiter.enabled = False
    yield
    limiter.enabled = previous
