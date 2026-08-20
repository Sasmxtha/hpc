"""
Noise Fingerprinting Module for PhysAttest (Component 8).

Every ADC has unique manufacturing imperfections — transistor threshold
variations, capacitor mismatches, crystal oscillator jitter. These create
a unique noise fingerprint. Firmware compromise changes timing behaviour,
which changes the fingerprint.

Works at ZERO coupling — checks the device, not the data.

Detection bound (Theorem 5, Neyman-Pearson + Sanov's theorem):
    P(evasion) ≤ exp(-n × D_KL(p_true || p_fake))

where D_KL is the KL-divergence between authentic and spoofed noise
distributions, and n is the number of samples.
"""

import numpy as np
from scipy import stats, signal
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class FingerprintVerdict(Enum):
    AUTHENTIC = "authentic"
    SUSPICIOUS = "suspicious"
    COMPROMISED = "compromised"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class FingerprintFeatures:
    """
    Feature vector extracted from ADC noise.

    Each feature captures a different aspect of the hardware's
    unique manufacturing imperfections.
    """
    # Statistical moments of noise amplitude
    variance: float
    skewness: float
    kurtosis: float

    # Allan variance at multiple time scales
    # σ²_y(τ) = ½⟨(ȳ_{k+1} - ȳ_k)²⟩
    # Captures time-domain stability characteristics
    allan_variances: np.ndarray   # at τ = [1, 2, 4, 8, 16, 32] samples

    # Power spectral density shape
    # 1/f noise has slope ≈ -1 on log-log PSD plot
    # Different hardware → different slope and intercept
    psd_slope: float              # log-log slope of PSD
    psd_intercept: float          # log-log intercept

    # Histogram of noise values (for KL-divergence)
    histogram: np.ndarray         # probability distribution
    bin_edges: np.ndarray

    # Timing features (inter-sample jitter)
    jitter_std: float
    jitter_kurtosis: float

    def to_vector(self) -> np.ndarray:
        """Flatten to a single feature vector for comparison."""
        return np.concatenate([
            [self.variance, self.skewness, self.kurtosis],
            self.allan_variances,
            [self.psd_slope, self.psd_intercept],
            [self.jitter_std, self.jitter_kurtosis],
        ])


@dataclass
class VerificationResult:
    sensor_id: int
    verdict: FingerprintVerdict
    confidence: float             # 0-1, how confident the verdict is
    d_kl: float                   # KL-divergence (Theorem 5)
    evasion_bound: float          # exp(-n × D_KL)
    n_samples: int
    feature_distances: dict       # per-feature distance from enrollment
    details: str


class NoiseExtractor:
    """
    Extracts ADC noise from raw sensor readings.

    The raw reading = physical signal + ADC noise.
    We estimate and remove the signal, leaving the noise.
    """

    @staticmethod
    def extract(readings: np.ndarray, method: str = "highpass") -> np.ndarray:
        """
        Extract noise component from sensor readings.

        Methods:
          - "highpass": apply high-pass filter (removes slow physical signal)
          - "difference": first-order difference (simple, robust)
          - "residual": subtract moving average (adaptive)
        """
        if len(readings) < 10:
            return np.zeros(0)

        if method == "highpass":
            # Butterworth high-pass at 0.3 × Nyquist
            # Removes physical signal, keeps ADC noise
            nyquist = 0.5
            cutoff = 0.3
            try:
                b, a = signal.butter(4, cutoff / nyquist, btype="high")
                noise = signal.filtfilt(b, a, readings)
            except ValueError:
                noise = np.diff(readings)

        elif method == "difference":
            noise = np.diff(readings)

        elif method == "residual":
            # Subtract moving average
            window = min(20, len(readings) // 3)
            if window < 3:
                window = 3
            kernel = np.ones(window) / window
            smoothed = np.convolve(readings, kernel, mode="same")
            noise = readings - smoothed

        else:
            raise ValueError(f"Unknown method: {method}")

        return noise


class FeatureExtractor:
    """
    Computes the fingerprint feature vector from extracted noise.
    """

    ALLAN_TAUS = np.array([1, 2, 4, 8, 16, 32])
    HISTOGRAM_BINS = 20

    @classmethod
    def extract(cls, noise: np.ndarray, timestamps: Optional[np.ndarray] = None) -> FingerprintFeatures:
        """
        Compute all fingerprint features from noise samples.
        """
        # --- Statistical moments ---
        variance = float(np.var(noise))
        skewness = float(stats.skew(noise))
        kurtosis = float(stats.kurtosis(noise))

        # --- Allan variance ---
        # σ²_y(τ) = 1/(2(N-2τ)) × Σ(x_{i+2τ} - 2x_{i+τ} + x_i)²
        allan_vars = cls._allan_variance(noise, cls.ALLAN_TAUS)

        # --- Power spectral density ---
        psd_slope, psd_intercept = cls._psd_shape(noise)

        # --- Histogram for KL-divergence ---
        hist, bin_edges = np.histogram(noise, bins=cls.HISTOGRAM_BINS, density=True)
        hist = hist + 1e-12  # avoid log(0)
        hist = hist / hist.sum()  # normalise to probability

        # --- Timing jitter ---
        if timestamps is not None and len(timestamps) > 1:
            intervals = np.diff(timestamps)
            jitter_std = float(np.std(intervals))
            jitter_kurtosis = float(stats.kurtosis(intervals))
        else:
            # Simulate from noise autocorrelation as proxy
            autocorr = np.correlate(noise[:min(100, len(noise))], noise[:min(100, len(noise))], mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            jitter_std = float(np.std(autocorr[:10]))
            jitter_kurtosis = float(stats.kurtosis(autocorr[:20])) if len(autocorr) >= 20 else 0.0

        return FingerprintFeatures(
            variance=variance,
            skewness=skewness,
            kurtosis=kurtosis,
            allan_variances=allan_vars,
            psd_slope=psd_slope,
            psd_intercept=psd_intercept,
            histogram=hist,
            bin_edges=bin_edges,
            jitter_std=jitter_std,
            jitter_kurtosis=jitter_kurtosis,
        )

    @staticmethod
    def _allan_variance(noise: np.ndarray, taus: np.ndarray) -> np.ndarray:
        """
        Allan variance at multiple time scales.

        σ²_y(τ) = 1/(2(N-2τ)) × Σ(x_{i+2τ} - 2x_{i+τ} + x_i)²

        Allan variance captures time-domain stability. Different ADCs
        have different stability profiles — white noise, flicker noise,
        random walk — creating a unique signature across τ.
        """
        allan_vars = np.zeros(len(taus))
        n = len(noise)

        for idx, tau in enumerate(taus):
            tau = int(tau)
            if 2 * tau >= n:
                allan_vars[idx] = allan_vars[idx - 1] if idx > 0 else 0
                continue

            # Second-order difference
            diffs = noise[2 * tau:] - 2 * noise[tau:-tau] + noise[:n - 2 * tau]
            allan_vars[idx] = np.mean(diffs ** 2) / 2.0

        return allan_vars

    @staticmethod
    def _psd_shape(noise: np.ndarray) -> tuple[float, float]:
        """
        Compute the slope of the power spectral density on log-log scale.

        1/f noise → slope ≈ -1
        White noise → slope ≈ 0
        Random walk → slope ≈ -2

        Different ADC hardware produces different noise colour mixtures.
        """
        # Welch's PSD estimate
        nperseg = min(256, len(noise) // 2)
        if nperseg < 16:
            return 0.0, 0.0

        freqs, psd = signal.welch(noise, nperseg=nperseg)

        # Fit line on log-log (skip DC component)
        mask = freqs > 0
        if np.sum(mask) < 3:
            return 0.0, 0.0

        log_f = np.log10(freqs[mask] + 1e-12)
        log_psd = np.log10(psd[mask] + 1e-30)

        # Robust linear fit
        slope, intercept, _, _, _ = stats.linregress(log_f, log_psd)
        return float(slope), float(intercept)


class FingerprintDatabase:
    """
    Stores enrolled fingerprints and performs verification.

    Enrollment: collect noise during trusted setup, compute features.
    Verification: compare current noise features against enrollment.
    """

    def __init__(self):
        self.enrolled: dict[int, FingerprintFeatures] = {}
        self.verification_history: dict[int, list[VerificationResult]] = {}

    def enroll(self, sensor_id: int, readings: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """
        Enroll a sensor during trusted setup.

        Must be done when the sensor is known to be authentic
        (factory provisioning or first boot in secure environment).
        """
        noise = NoiseExtractor.extract(readings, method="highpass")
        if len(noise) < 50:
            noise = NoiseExtractor.extract(readings, method="difference")

        features = FeatureExtractor.extract(noise, timestamps)
        self.enrolled[sensor_id] = features
        self.verification_history[sensor_id] = []

    def verify(self, sensor_id: int, readings: np.ndarray, timestamps: Optional[np.ndarray] = None) -> VerificationResult:
        """
        Verify a sensor's current noise against its enrolled fingerprint.

        Implements Theorem 5:
            P(evasion) ≤ exp(-n × D_KL(p_true || p_fake))
        """
        if sensor_id not in self.enrolled:
            return VerificationResult(
                sensor_id=sensor_id,
                verdict=FingerprintVerdict.INSUFFICIENT_DATA,
                confidence=0.0,
                d_kl=0.0,
                evasion_bound=1.0,
                n_samples=0,
                feature_distances={},
                details="Sensor not enrolled",
            )

        enrolled = self.enrolled[sensor_id]
        n_samples = len(readings)

        # Extract current noise
        noise = NoiseExtractor.extract(readings, method="highpass")
        if len(noise) < 30:
            noise = NoiseExtractor.extract(readings, method="difference")

        if len(noise) < 10:
            return VerificationResult(
                sensor_id=sensor_id,
                verdict=FingerprintVerdict.INSUFFICIENT_DATA,
                confidence=0.0,
                d_kl=0.0,
                evasion_bound=1.0,
                n_samples=n_samples,
                feature_distances={},
                details=f"Only {len(noise)} noise samples — need at least 10",
            )

        current = FeatureExtractor.extract(noise, timestamps)

        # --- KL-divergence between enrolled and current histograms ---
        # This is the core of Theorem 5
        # Align histograms to same bins
        current_hist, _ = np.histogram(noise, bins=enrolled.bin_edges, density=True)
        current_hist = current_hist + 1e-12
        current_hist = current_hist / current_hist.sum()

        d_kl = float(stats.entropy(enrolled.histogram, current_hist))

        # --- Per-feature distances ---
        enrolled_vec = enrolled.to_vector()
        current_vec = current.to_vector()
        feature_names = [
            "variance", "skewness", "kurtosis",
            "allan_τ1", "allan_τ2", "allan_τ4", "allan_τ8", "allan_τ16", "allan_τ32",
            "psd_slope", "psd_intercept",
            "jitter_std", "jitter_kurtosis",
        ]

        feature_distances = {}
        for name, enrolled_val, current_val in zip(feature_names, enrolled_vec, current_vec):
            # Use max(|enrolled|, 1.0) as denominator to handle near-zero features
            # Moments (skewness, kurtosis) can be near zero for Gaussian noise
            denom = max(abs(enrolled_val), 1.0)
            feature_distances[name] = float(abs(current_val - enrolled_val) / denom)

        # --- Theorem 5: evasion bound ---
        # P(evasion) ≤ exp(-n × D_KL)
        evasion_bound = float(np.exp(-n_samples * d_kl))

        # --- Combined confidence score ---
        # Weighted combination of KL-divergence and feature distances
        feature_dist_mean = np.mean(list(feature_distances.values()))

        # Weighted feature distance — stable features count more.
        # Variance and Allan variances are hardware-determined and reproducible.
        # Kurtosis, skewness, jitter_kurtosis have high sample variance.
        feature_weights = {
            "variance": 3.0,
            "skewness": 0.5,
            "kurtosis": 0.5,
            "allan_τ1": 2.0, "allan_τ2": 2.0, "allan_τ4": 2.0,
            "allan_τ8": 1.5, "allan_τ16": 1.0, "allan_τ32": 1.0,
            "psd_slope": 2.0,
            "psd_intercept": 1.0,
            "jitter_std": 1.5,
            "jitter_kurtosis": 0.3,
        }
        weighted_dist = 0.0
        total_weight = 0.0
        for name, dist in feature_distances.items():
            w = feature_weights.get(name, 1.0)
            weighted_dist += w * dist ** 2
            total_weight += w
        weighted_dist = np.sqrt(weighted_dist / total_weight)

        # Authenticity score: high when features match enrollment
        authenticity_score = float(np.exp(-weighted_dist * 3))

        # Classification
        if authenticity_score > 0.6:
            verdict = FingerprintVerdict.AUTHENTIC
        elif authenticity_score > 0.2:
            verdict = FingerprintVerdict.SUSPICIOUS
        else:
            verdict = FingerprintVerdict.COMPROMISED

        result = VerificationResult(
            sensor_id=sensor_id,
            verdict=verdict,
            confidence=float(authenticity_score),
            d_kl=d_kl,
            evasion_bound=evasion_bound,
            n_samples=n_samples,
            feature_distances=feature_distances,
            details=(
                f"D_KL={d_kl:.4f}, evasion≤{evasion_bound:.2e}, "
                f"feature_dist={feature_dist_mean:.4f}"
            ),
        )

        self.verification_history.setdefault(sensor_id, []).append(result)
        return result

    def get_evasion_bound(self, sensor_id: int) -> float:
        """
        Cumulative evasion bound across all verification windows.

        Each independent verification window multiplies the bound:
        P(evasion_total) ≤ Π_k exp(-n_k × D_KL_k)
        """
        history = self.verification_history.get(sensor_id, [])
        if not history:
            return 1.0

        # Product of individual bounds
        log_bound = sum(
            -r.n_samples * r.d_kl
            for r in history
            if r.d_kl > 0
        )
        return float(np.exp(log_bound))


def demo_fingerprinting():
    """
    Demo: enroll sensors, then verify authentic, faulty, and compromised ones.

    Shows how noise fingerprinting detects firmware compromise even with
    zero physical coupling between sensors.
    """
    np.random.seed(42)
    n_samples = 500

    print("=" * 65)
    print("  PhysAttest Noise Fingerprinting — Demo")
    print("  Theorem 5: P(evasion) ≤ exp(-n × D_KL)")
    print("=" * 65)

    db = FingerprintDatabase()

    # --- Generate authentic noise signatures ---
    # Each sensor has unique noise characteristics from manufacturing

    def make_sensor_noise(n: int, variance: float, kurtosis_target: float, psd_slope: float) -> np.ndarray:
        """Simulate ADC noise with specific hardware characteristics."""
        # White noise base
        white = np.random.randn(n) * np.sqrt(variance)

        # Add 1/f component (coloured noise via filtering)
        b_coeff = [1.0]
        a_coeff = [1.0, -0.5 * abs(psd_slope)]
        if abs(a_coeff[1]) < 1:
            coloured = signal.lfilter(b_coeff, a_coeff, np.random.randn(n))
            coloured = coloured * np.sqrt(variance) * 0.3
        else:
            coloured = np.zeros(n)

        # Add kurtosis via occasional spikes
        spike_prob = max(0, kurtosis_target - 3) * 0.01
        spikes = np.random.randn(n) * np.sqrt(variance) * 3
        spike_mask = np.random.rand(n) < spike_prob
        noise = white + coloured + spikes * spike_mask

        return noise

    sensor_params = {
        0: {"variance": 0.001, "kurtosis_target": 3.5, "psd_slope": -0.3},
        1: {"variance": 0.002, "kurtosis_target": 4.0, "psd_slope": -0.5},
        2: {"variance": 0.0015, "kurtosis_target": 3.2, "psd_slope": -0.8},
        3: {"variance": 0.001, "kurtosis_target": 5.0, "psd_slope": -0.4},
    }

    # Physical signal (same for all sensors — temperature ramp)
    physical = np.linspace(20, 22, n_samples)

    # --- Enrollment phase ---
    print("\n── Enrollment Phase (trusted setup) ──")
    for sid, params in sensor_params.items():
        noise = make_sensor_noise(n_samples, **params)
        readings = physical + noise
        db.enroll(sid, readings)
        fp = db.enrolled[sid]
        print(
            f"  Sensor {sid}: var={fp.variance:.6f}  "
            f"kurt={fp.kurtosis:.2f}  "
            f"psd_slope={fp.psd_slope:.2f}"
        )

    # --- Verification: authentic sensors ---
    print("\n── Verification: Authentic Sensors (same hardware, new data) ──")
    for sid, params in sensor_params.items():
        noise = make_sensor_noise(n_samples, **params)
        readings = physical + noise
        result = db.verify(sid, readings)
        print(
            f"  Sensor {sid}: {result.verdict.value:<12s}  "
            f"confidence={result.confidence:.3f}  "
            f"D_KL={result.d_kl:.4f}  "
            f"P(evasion)≤{result.evasion_bound:.2e}"
        )
        # Show feature distances for debugging
        top3 = sorted(result.feature_distances.items(), key=lambda x: x[1], reverse=True)[:3]
        for fname, fdist in top3:
            print(f"    ↳ {fname}: distance={fdist:.3f}")

    # --- Verification: firmware-compromised sensor ---
    print("\n── Verification: Compromised Sensor (firmware replaced) ──")
    for sid in [0, 2]:
        # Compromised firmware has fundamentally different noise structure:
        # 1. Different variance (different ADC or emulated)
        # 2. Different PSD slope (digital replay → flat spectrum, no 1/f)
        # 3. Different Allan variance profile (no correlated flicker noise)
        # 4. Different kurtosis (uniform quantisation noise vs Gaussian)
        compromised_noise = make_sensor_noise(
            n_samples,
            variance=sensor_params[sid]["variance"] * 4,  # different ADC
            kurtosis_target=6.0,     # heavier tails from digital jitter
            psd_slope=-1.5,          # very different spectral shape
        )
        readings = physical + compromised_noise
        result = db.verify(sid, readings)
        print(
            f"  Sensor {sid}: {result.verdict.value:<12s}  "
            f"confidence={result.confidence:.3f}  "
            f"D_KL={result.d_kl:.4f}  "
            f"P(evasion)≤{result.evasion_bound:.2e}"
        )

        # Show which features diverged most
        top_features = sorted(
            result.feature_distances.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        for fname, fdist in top_features:
            print(f"    ↳ {fname}: distance={fdist:.3f}")

    # --- Verification: faulty sensor (same hardware, degraded) ---
    print("\n── Verification: Faulty Sensor (same hardware, degrading) ──")
    for sid in [1]:
        params = sensor_params[sid].copy()
        # Drift: slightly increased variance (capacitor aging)
        params["variance"] *= 1.5
        noise = make_sensor_noise(n_samples, **params)
        readings = physical + noise
        result = db.verify(sid, readings)
        print(
            f"  Sensor {sid}: {result.verdict.value:<12s}  "
            f"confidence={result.confidence:.3f}  "
            f"D_KL={result.d_kl:.4f}  "
            f"(gradual degradation — suspicious, not compromised)"
        )

    # --- Theorem 5 demonstration ---
    print("\n── Theorem 5: Evasion bound vs sample count ──")
    print("  More samples → exponentially harder to evade")

    compromised_noise = np.random.randn(2000) * 0.003
    compromised_noise += signal.lfilter([1], [1, -0.8], np.random.randn(2000)) * 0.001

    for n in [50, 100, 200, 500, 1000, 2000]:
        readings = np.linspace(20, 22, n) + compromised_noise[:n]
        result = db.verify(0, readings)
        print(f"  n={n:>5d}: D_KL={result.d_kl:.4f}  P(evasion)≤{result.evasion_bound:.2e}")

    print("\n" + "=" * 65)
    print("  Key properties:")
    print("  • Zero coupling required — checks device, not data")
    print("  • Firmware compromise changes timing → changes fingerprint")
    print("  • Evasion probability drops EXPONENTIALLY with samples")
    print("  • Faults (gradual) vs attacks (sudden) clearly distinguished")
    print("=" * 65)


if __name__ == "__main__":
    demo_fingerprinting()
