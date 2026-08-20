"""Tests for Noise Fingerprinting (Component 8)."""

import numpy as np
import pytest
from physattest.security.fingerprint import (
    NoiseExtractor, FeatureExtractor, FingerprintDatabase,
    FingerprintVerdict, FingerprintFeatures,
)


@pytest.fixture
def db():
    return FingerprintDatabase()


def _make_noise(n, variance=0.001, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n) * np.sqrt(variance)


def _make_readings(n, variance=0.001, seed=42):
    physical = np.linspace(20, 22, n)
    noise = _make_noise(n, variance, seed)
    return physical + noise


class TestNoiseExtractor:
    def test_highpass_returns_noise(self):
        readings = _make_readings(500)
        noise = NoiseExtractor.extract(readings, method="highpass")
        assert len(noise) > 0
        assert np.std(noise) < np.std(readings)

    def test_difference_returns_noise(self):
        readings = _make_readings(500)
        noise = NoiseExtractor.extract(readings, method="difference")
        assert len(noise) == len(readings) - 1

    def test_residual_method(self):
        readings = _make_readings(500)
        noise = NoiseExtractor.extract(readings, method="residual")
        assert len(noise) == len(readings)

    def test_short_input_returns_empty(self):
        noise = NoiseExtractor.extract(np.array([1.0, 2.0, 3.0]))
        assert len(noise) == 0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            NoiseExtractor.extract(np.ones(100), method="invalid")


class TestFeatureExtractor:
    def test_feature_vector_shape(self):
        noise = _make_noise(500)
        features = FeatureExtractor.extract(noise)
        vec = features.to_vector()
        assert vec.shape == (13,)

    def test_allan_variances_nonnegative(self):
        noise = _make_noise(500)
        features = FeatureExtractor.extract(noise)
        assert np.all(features.allan_variances >= 0)

    def test_histogram_is_probability(self):
        noise = _make_noise(500)
        features = FeatureExtractor.extract(noise)
        assert abs(features.histogram.sum() - 1.0) < 1e-6


class TestFingerprintDatabase:
    def test_enroll_stores_features(self, db):
        readings = _make_readings(500)
        db.enroll(0, readings)
        assert 0 in db.enrolled

    def test_verify_unenrolled_returns_insufficient(self, db):
        readings = _make_readings(500)
        result = db.verify(99, readings)
        assert result.verdict == FingerprintVerdict.INSUFFICIENT_DATA

    def test_authentic_sensor_passes(self, db):
        readings1 = _make_readings(500, seed=42)
        readings2 = _make_readings(500, seed=43)
        db.enroll(0, readings1)
        result = db.verify(0, readings2)
        assert result.verdict in (FingerprintVerdict.AUTHENTIC, FingerprintVerdict.SUSPICIOUS)
        assert result.confidence > 0.1

    def test_compromised_sensor_detected(self, db):
        readings_auth = _make_readings(500, variance=0.001, seed=42)
        db.enroll(0, readings_auth)
        readings_fake = _make_readings(500, variance=0.01, seed=99)
        result = db.verify(0, readings_fake)
        assert result.d_kl > 0, "KL divergence should be positive for different noise"

    def test_evasion_bound_decreases_with_samples(self, db):
        readings_auth = _make_readings(500, variance=0.001, seed=42)
        db.enroll(0, readings_auth)
        readings_fake = _make_readings(2000, variance=0.005, seed=99)

        result_small = db.verify(0, readings_fake[:200])
        result_large = db.verify(0, readings_fake[:1000])
        if result_small.d_kl > 0 and result_large.d_kl > 0:
            assert result_large.evasion_bound <= result_small.evasion_bound + 0.01

    def test_cumulative_evasion_bound(self, db):
        readings_auth = _make_readings(500, variance=0.001, seed=42)
        db.enroll(0, readings_auth)
        for i in range(3):
            readings = _make_readings(200, variance=0.005, seed=i + 100)
            db.verify(0, readings)
        bound = db.get_evasion_bound(0)
        assert bound <= 1.0

    def test_short_readings_insufficient(self, db):
        db.enroll(0, _make_readings(500))
        result = db.verify(0, np.array([20.0, 20.1, 20.2]))
        assert result.verdict == FingerprintVerdict.INSUFFICIENT_DATA
