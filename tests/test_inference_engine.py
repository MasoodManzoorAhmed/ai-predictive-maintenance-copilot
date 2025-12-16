import os
import pytest

# GitHub Actions sets CI=true
if os.getenv("CI", "").lower() == "true":
    pytest.skip("Skipping inference/model-artifact tests in CI build gate.", allow_module_level=True)
