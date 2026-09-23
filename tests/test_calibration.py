import numpy as np

from verdict.calibration import ConformalCalibrator, TemperatureScaler, expected_calibration_error


def _synthetic(n=4000, k=10, sharpness=3.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), y] += 2.0
    return logits * sharpness, y  # over-confident by construction


def test_temperature_reduces_ece():
    logits, y = _synthetic()
    raw = TemperatureScaler(1.0).transform(logits)
    ts = TemperatureScaler().fit(logits[:2000], y[:2000])
    cal = ts.transform(logits[2000:])
    assert ts.temperature > 1.2
    assert expected_calibration_error(cal, y[2000:]) < expected_calibration_error(raw[2000:], y[2000:]) / 2


def test_conformal_coverage():
    logits, y = _synthetic(sharpness=1.0)
    logits[np.arange(len(y)), y] += 3.0  # an easier, well-separated task
    p = TemperatureScaler().fit(logits[:2000], y[:2000]).transform(logits)
    cc = ConformalCalibrator(alpha=0.1).fit(p[2000:3000], y[2000:3000])
    sets = cc.prediction_set(p[3000:])
    covered = np.mean([yy in s for yy, s in zip(y[3000:], sets)])
    assert covered >= 0.88  # 1 - alpha minus finite-sample slack
    assert np.mean([len(s) == 1 for s in sets]) > 0.5  # still commits often


def test_roundtrip():
    cc = ConformalCalibrator(alpha=0.05, qhat=0.9)
    assert ConformalCalibrator.from_dict(cc.to_dict()).qhat == 0.9
    assert TemperatureScaler.from_dict(TemperatureScaler(2.5).to_dict()).temperature == 2.5
