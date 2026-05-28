
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TOOLS = os.path.join(ROOT, 'tools')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from sam_extractor import build_object_prior_maps  # noqa: E402


def test_object_prior_maps_empty_when_classes_absent():
    probs = np.zeros((12, 16, 3), dtype=np.float32)
    out = build_object_prior_maps(probs, ['rail', 'ballast', 'sky'])

    assert out['person']['detected_count'] == 0
    assert out['vehicle']['detected_count'] == 0
    assert out['person']['weight'].shape == (12, 16)
    assert float(out['person']['weight'].max()) == 0.0
    assert float(out['vehicle']['dist'].max()) == 1.0


def test_object_prior_maps_detect_vehicle_and_person_channels():
    probs = np.zeros((32, 40, 4), dtype=np.float32)
    classes = ['rail', 'person', 'vehicle', 'sky']
    probs[5:12, 10:14, 1] = 0.9
    probs[18:24, 20:31, 2] = 0.8

    out = build_object_prior_maps(probs, classes, min_prob=0.2)

    assert out['person']['detected_count'] >= 1
    assert out['vehicle']['detected_count'] >= 1
    assert out['person']['weight'][8, 12] > 0.5
    assert out['vehicle']['weight'][20, 24] > 0.5
    assert out['person']['dist'][8, 12] < 0.05
    assert out['vehicle']['dist'][20, 24] < 0.05
