from __future__ import annotations

import pytest
from facial_recognizer.inference import facial_recognizer


def test_facial_recognizer():
    with pytest.raises(NotImplementedError, match='The facial_recognizer function is not yet implemented.'):
        facial_recognizer()
