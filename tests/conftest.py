from hypothesis import settings
import pytest

# A profile for quick demo
settings.register_profile("dev", max_examples=1, deadline=None)

# A profile for quick local dev
settings.register_profile("dev", max_examples=5, deadline=None)

# A profile for heavy stress-testing (e.g., CI or nightly runs)
settings.register_profile("stress", max_examples=100, deadline=None)

# Set a default
settings.load_profile("dev")

def pytest_addoption(parser):
    # Add flags for all your matmul parameters
    parser.addoption("--M", action="store", default=None, type=int)
    parser.addoption("--N", action="store", default=None, type=int)
    parser.addoption("--K", action="store", default=None, type=int)
    parser.addoption("--warps", action="store", default=4, type=int)
    parser.addoption("--stages", action="store", default=2, type=int)
    parser.addoption("--trans", action="store_true", default=False)
