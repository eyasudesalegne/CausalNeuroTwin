"""Generic numerical controls, never private inventories."""

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def options():
    p = Path(__file__).resolve().parents[3] / "configs/scientific_validation.synthetic.example.json"
    return json.loads(p.read_text())


@pytest.fixture
def connectomes():
    a = np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]], float)
    b = np.array([[0, 3, 4], [3, 0, 1], [4, 1, 0]], float)
    return a, b, np.where(a > 0, 20.0, 0), np.where(b > 0, 30.0, 0)
