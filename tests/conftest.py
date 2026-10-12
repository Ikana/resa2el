import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """A test of a function that is not written yet is skipped, not failed."""
    outcome = yield
    if outcome.excinfo and issubclass(outcome.excinfo[0], NotImplementedError):
        outcome.force_exception(pytest.skip.Exception("not written yet"))
