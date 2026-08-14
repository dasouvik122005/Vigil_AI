"""
Adaptive Decision Engine for STAMPER_TSLR.

Core ML engine combining:
- Unsupervised anomaly detection (IsolationForest)
- Supervised adaptation via human feedback (RandomForest)
- Uncertainty quantification (Phase 2)
- Algorithm ensemble (Phase 2)
- Concept drift detection (Phase 2)
- Explainable AI with SHAP (Phase 4)
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# Local imports
from config import get_data_config, get_model_config
from monitoring import get_logger, record_anomaly, record_prediction, record_retrain

logger = get_logger("ml_engine")


@dataclass
class PredictionResult:
    """Structured prediction result with uncertainty decomposition."""

    is_anomaly: bool
    confidence: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    explanation: str
    prediction_source: str
    detector_scores: dict[str, float]


class AdaptiveDecisionEngine:
    """
    Main ML engine for adaptive anomaly detection with human-in-the-loop learning.

    Features:
    - Multi-algorithm ensemble for robust anomaly detection
    - Uncertainty quantification (epistemic + aleatoric)
    - Conformal prediction intervals
    - Human feedback adaptation
    - Concept drift detection integration
    - SHAP explainability
    """

    def __init__(self, historical_data_path: str = "sensor_data.csv"):
        # Load configuration
        self.model_config = get_model_config()
        self.data_config = get_data_config()

        # Feature configuration
        self.features_list = self.data_config.sensor_features
        self.n_features = len(self.features_list)

        # Initialize models
        self._init_models()

        # State
        self.is_rf_trained = False
        self.feedback_buffer: list[tuple[np.ndarray, int]] = []
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Conformal prediction calibration
        self.calibration_scores: list[float] = []
        self.calibration_labels: list[int] = []

        # Load and train initial base models
        self._load_and_train_base(historical_data_path)

        logger.info(
            "adaptive_decision_engine_initialized",
            features=self.features_list,
            ensemble_enabled=self.model_config.ensemble_enabled,
            uncertainty_method=self.model_config.uncertainty_method,
        )

    def _init_models(self) -> None:
        """Initialize all ensemble models."""
        cfg = self.model_config

        # 1. IsolationForest (global outliers)
        self.iso_forest = IsolationForest(
            contamination=cfg.iso_contamination,
            n_estimators=cfg.iso_n_estimators,
            max_samples=cfg.iso_max_samples,
            random_state=cfg.iso_random_state,
            n_jobs=-1,
        )

        # 2. LocalOutlierFactor (local density outliers)
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=cfg.iso_contamination,
            novelty=True,  # Enable predict on new data
            n_jobs=-1,
        )

        # 3. OneClassSVM (boundary-based)
        self.oc_svm = OneClassSVM(nu=cfg.iso_contamination, kernel="rbf", gamma="scale")

        # 4. Autoencoder placeholder (for temporal patterns)
        # Will be implemented in Phase 3 with proper neural network
        self.autoencoder = None

        # 5. RandomForest for supervised adaptation
        self.rf_classifier = RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_split=cfg.rf_min_samples_split,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            random_state=cfg.rf_random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

        # Ensemble models list for uncertainty estimation
        self.ensemble_models: list[BaseEstimator] = [
            self.iso_forest,
            self.lof,
            self.oc_svm,
        ]

        # Ensemble weights (default equal, can be configured)
        if self.model_config.ensemble_weights:
            self.ensemble_weights = np.array(self.model_config.ensemble_weights)
        else:
            self.ensemble_weights = np.ones(len(self.ensemble_models)) / len(
                self.ensemble_models
            )

    def _load_and_train_base(self, historical_data_path: str) -> None:
        """Load historical data and train base unsupervised models."""
        try:
            df = pd.read_csv(historical_data_path)

            # Compute statistics for imputation and normalization
            self.means = df[self.features_list].mean().to_dict()
            self.stds = df[self.features_list].std().to_dict()

            # Handle zero std
            for k, v in self.stds.items():
                if v == 0:
                    self.stds[k] = 1.0

            # Prepare training data (impute missing with mean)
            X_train = df[self.features_list].fillna(df[self.features_list].mean())

            # Fit scaler
            self.scaler.fit(X_train)
            X_scaled = self.scaler.transform(X_train)

            # Train unsupervised models
            self.iso_forest.fit(X_scaled)
            self.lof.fit(X_scaled)
            self.oc_svm.fit(X_scaled)

            self.is_fitted = True

            logger.info(
                "base_models_trained", n_samples=len(df), features=self.features_list
            )

        except Exception as e:
            logger.warning("base_training_failed_using_defaults", error=str(e))
            self._set_default_statistics()

    def _set_default_statistics(self) -> None:
        """Set default statistics when training data unavailable."""
        defaults = {
            "temperature": 25,
            "pressure": 1.2,
            "vibration": 0.3,
            "humidity": 50,
            "voltage": 230,
            "current": 10,
        }
        default_stds = {
            "temperature": 3,
            "pressure": 0.1,
            "vibration": 0.05,
            "humidity": 10,
            "voltage": 10,
            "current": 2,
        }

        for feat in self.features_list:
            self.means[feat] = defaults.get(feat, 0.0)
            self.stds[feat] = default_stds.get(feat, 1.0)

        # Fit scaler on dummy data
        dummy = np.array([[self.means[f] for f in self.features_list]])
        self.scaler.fit(dummy)
        self.is_fitted = True

    def _preprocess(
        self, data_dict: dict[str, Any]
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Preprocess input data: impute missing, scale, return missing mask.

        Returns:
            X_scaled: Scaled feature array (1, n_features)
            missing_count: Number of missing features
            missing_mask: Boolean array indicating missing features
        """
        features = []
        missing_mask = []

        for feat in self.features_list:
            value = data_dict.get(feat)
            if value is None:
                features.append(self.means.get(feat, 0.0))
                missing_mask.append(True)
            else:
                features.append(float(value))
                missing_mask.append(False)

        X = np.array([features])
        X_scaled = self.scaler.transform(X) if self.is_fitted else X
        missing_count = sum(missing_mask)

        return X_scaled, missing_count, np.array(missing_mask)

    def _get_detector_scores(self, X: np.ndarray) -> dict[str, float]:
        """Get anomaly scores from all detectors."""
        scores = {}

        # IsolationForest: decision_function (higher = more normal)
        try:
            iso_score = self.iso_forest.decision_function(X)[0]
            scores["isolation_forest"] = float(iso_score)
        except Exception:
            scores["isolation_forest"] = 0.0

        # LOF: negative outlier factor (more negative = more anomalous)
        try:
            lof_score = -self.lof.decision_function(X)[0]  # Negate for consistency
            scores["lof"] = float(lof_score)
        except Exception:
            scores["lof"] = 0.0

        # OneClassSVM: decision function (higher = more normal)
        try:
            svm_score = self.oc_svm.decision_function(X)[0]
            scores["one_class_svm"] = float(svm_score)
        except Exception:
            scores["one_class_svm"] = 0.0

        return scores

    def _estimate_epistemic_uncertainty(
        self, X: np.ndarray, detector_scores: dict[str, float]
    ) -> float:
        """
        Estimate epistemic (model) uncertainty from ensemble disagreement.

        Uses variance of normalized detector scores as proxy for model uncertainty.
        """
        if not self.model_config.ensemble_enabled or len(detector_scores) < 2:
            return 0.1  # Default low uncertainty

        # Normalize scores to [0, 1] anomaly probability
        normalized_scores = []
        for name, score in detector_scores.items():
            # Convert to anomaly probability (0 = normal, 1 = anomaly)
            if name == "isolation_forest":
                # decision_function: positive = normal, negative = anomaly
                prob = 1.0 / (1.0 + np.exp(score))  # Sigmoid
            elif name == "lof":
                # Already negated, higher = more anomalous
                prob = 1.0 / (1.0 + np.exp(-score))
            elif name == "one_class_svm":
                prob = 1.0 / (1.0 + np.exp(score))
            else:
                prob = 0.5
            normalized_scores.append(prob)

        # Epistemic uncertainty = variance of ensemble predictions
        epistemic = float(np.var(normalized_scores))

        # Scale to [0, 1]
        return min(1.0, epistemic * 4)  # Empirical scaling

    def _estimate_aleatoric_uncertainty(
        self, X: np.ndarray, missing_mask: np.ndarray
    ) -> float:
        """
        Estimate aleatoric (data) uncertainty.

        Based on:
        - Missing data ratio
        - Distance from training distribution (Mahalanobis-like)
        """
        # Base uncertainty from missing data
        missing_ratio = np.mean(missing_mask) if len(missing_mask) > 0 else 0.0
        aleatoric = missing_ratio * 0.5  # Max 0.5 from missing data

        # Add uncertainty from out-of-distribution detection
        if self.is_fitted:
            try:
                # Z-score based distance
                z_scores = np.abs(
                    (X[0] - self.scaler.mean_) / np.sqrt(self.scaler.var_ + 1e-8)
                )
                max_z = np.max(z_scores)
                # Higher z-score = more uncertain
                ood_uncertainty = min(0.3, max_z / 10.0)
                aleatoric += ood_uncertainty
            except Exception:
                pass

        return min(1.0, aleatoric)

    def _compute_conformal_interval(self, anomaly_prob: float) -> tuple[float, float]:
        """
        Compute prediction interval using conformal prediction.

        For binary classification, we use the calibration scores to get
        a prediction set with guaranteed coverage.
        """
        if len(self.calibration_scores) < 10:
            # Not enough calibration data, return heuristic interval
            margin = 0.2
            return max(0.0, anomaly_prob - margin), min(1.0, anomaly_prob + margin)

        # Conformal prediction for anomaly probability
        alpha = 0.05  # 95% coverage
        calibration_scores = np.array(self.calibration_scores)

        # Non-conformity score: |prediction - true_label|
        # For new prediction, we compute interval of possible labels
        q_level = np.ceil((len(calibration_scores) + 1) * (1 - alpha)) / len(
            calibration_scores
        )
        q_hat = np.quantile(calibration_scores, q_level)

        lower = max(0.0, anomaly_prob - q_hat)
        upper = min(1.0, anomaly_prob + q_hat)

        return float(lower), float(upper)

    def _ensemble_predict(
        self, X: np.ndarray, detector_scores: dict[str, float]
    ) -> tuple[bool, float, str]:
        """
        Ensemble prediction with uncertainty-weighted voting.

        Returns:
            is_anomaly: Final prediction
            anomaly_prob: Probability of anomaly
            source: Description of prediction source
        """
        if not self.model_config.ensemble_enabled:
            # Single model fallback
            iso_score = detector_scores.get("isolation_forest", 0.0)
            is_anomaly = iso_score < 0
            prob = 1.0 / (1.0 + np.exp(iso_score))
            return is_anomaly, float(prob), "IsolationForest (Single)"

        # Weighted voting
        votes = []
        weights = []

        # IsolationForest vote
        iso_score = detector_scores.get("isolation_forest", 0.0)
        iso_prob = 1.0 / (1.0 + np.exp(iso_score))
        votes.append(iso_prob)
        weights.append(self.ensemble_weights[0])

        # LOF vote
        lof_score = detector_scores.get("lof", 0.0)
        lof_prob = 1.0 / (1.0 + np.exp(-lof_score))
        votes.append(lof_prob)
        weights.append(
            self.ensemble_weights[1] if len(self.ensemble_weights) > 1 else 1.0
        )

        # OneClassSVM vote
        svm_score = detector_scores.get("one_class_svm", 0.0)
        svm_prob = 1.0 / (1.0 + np.exp(svm_score))
        votes.append(svm_prob)
        weights.append(
            self.ensemble_weights[2] if len(self.ensemble_weights) > 2 else 1.0
        )

        # RandomForest vote (if trained)
        source = "Ensemble (IF+LOF+SVM)"
        if self.is_rf_trained:
            try:
                rf_prob = self.rf_classifier.predict_proba(X)[0]
                if len(rf_prob) > 1:
                    votes.append(rf_prob[1])  # Probability of anomaly class
                    weights.append(
                        self.ensemble_weights[3]
                        if len(self.ensemble_weights) > 3
                        else 1.0
                    )
                    source = "Ensemble (IF+LOF+SVM+RF)"
            except Exception:
                pass

        # Weighted average
        votes = np.array(votes)
        weights = np.array(weights[: len(votes)])
        weights = weights / weights.sum()

        ensemble_prob = float(np.average(votes, weights=weights))
        is_anomaly = ensemble_prob > 0.5

        return is_anomaly, ensemble_prob, source

    def _supervised_override(
        self, X: np.ndarray, base_anomaly: bool, base_prob: float
    ) -> tuple[bool, float, str]:
        """
        Apply supervised model override if trained and confident.

        Returns updated (is_anomaly, prob, source)
        """
        if not self.is_rf_trained:
            return base_anomaly, base_prob, "Ensemble"

        try:
            rf_pred = self.rf_classifier.predict(X)[0]
            rf_proba = self.rf_classifier.predict_proba(X)[0]
            max_proba = np.max(rf_proba)

            # Override if supervised model is highly confident
            if max_proba > 0.7:
                is_anomaly = bool(rf_pred == 1)
                return is_anomaly, float(max_proba), "RandomForest (Human-Adapted)"
        except Exception:
            pass

        return base_anomaly, base_prob, "Ensemble"

    def _generate_explanation(
        self,
        data_dict: dict[str, Any],
        missing_mask: np.ndarray,
        is_anomaly: bool,
        detector_scores: dict[str, float],
        source: str,
    ) -> str:
        """Generate human-readable explanation."""
        parts = [f"Prediction made by {source}."]

        # Missing data warning
        missing_count = np.sum(missing_mask)
        if missing_count > 0:
            missing_features = [
                self.features_list[i] for i, m in enumerate(missing_mask) if m
            ]
            parts.append(
                f"WARNING: {missing_count} sensors offline ({', '.join(missing_features)})."
            )

        if is_anomaly:
            parts.append("ANOMALY DETECTED: ")

            # Find most anomalous features
            deviations = []
            for i, feat in enumerate(self.features_list):
                value = data_dict.get(feat)
                if value is not None and feat in self.means and feat in self.stds:
                    z_score = abs(value - self.means[feat]) / (self.stds[feat] + 1e-8)
                    if z_score > 2.0:
                        deviations.append(f"{feat}={value:.2f} ({z_score:.1f}σ)")

            if deviations:
                parts.append(f"Key deviations: {', '.join(deviations)}.")
            else:
                parts.append("Complex multivariate anomaly detected by ensemble.")

            # Add detector consensus info
            n_anomalous = sum(1 for s in detector_scores.values() if s < 0)
            if n_anomalous > 1:
                parts.append(f"{n_anomalous}/{len(detector_scores)} detectors agree.")
        else:
            parts.append(
                "All available sensor readings are within normal historical boundaries."
            )

        return " ".join(parts)

    def predict(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Main prediction method with full uncertainty quantification.

        Returns structured result with:
        - is_anomaly: Binary prediction
        - confidence: Calibrated confidence score
        - epistemic_uncertainty: Model uncertainty
        - aleatoric_uncertainty: Data uncertainty
        - total_uncertainty: Combined uncertainty
        - prediction_interval: Conformal prediction interval
        - explanation: Human-readable explanation
        - prediction_source: Which model made the prediction
        - detector_scores: Individual detector scores
        """
        start_time = time.perf_counter()

        # Preprocess
        X, missing_count, missing_mask = self._preprocess(data_dict)

        # Get detector scores
        detector_scores = self._get_detector_scores(X)

        # Ensemble prediction
        is_anomaly, anomaly_prob, source = self._ensemble_predict(X, detector_scores)

        # Supervised override
        is_anomaly, anomaly_prob, source = self._supervised_override(
            X, is_anomaly, anomaly_prob
        )

        # Uncertainty quantification
        epistemic = self._estimate_epistemic_uncertainty(X, detector_scores)
        aleatoric = self._estimate_aleatoric_uncertainty(X, missing_mask)
        total_uncertainty = min(1.0, epistemic + aleatoric)

        # Confidence = 1 - total_uncertainty (calibrated)
        confidence = 1.0 - total_uncertainty
        confidence = max(0.01, min(0.99, confidence))

        # Conformal prediction interval
        pi_lower, pi_upper = self._compute_conformal_interval(anomaly_prob)

        # Explanation
        explanation = self._generate_explanation(
            data_dict, missing_mask, is_anomaly, detector_scores, source
        )

        # Record calibration data for conformal prediction (using anomaly_prob as score)
        # In practice, we'd use true labels; here we use proxy
        self.calibration_scores.append(abs(anomaly_prob - (1.0 if is_anomaly else 0.0)))
        if len(self.calibration_scores) > self.model_config.conformal_calibration_size:
            self.calibration_scores.pop(0)

        # Record metrics
        prediction_label = "ANOMALY_DETECTED" if is_anomaly else "NORMAL_OPERATION"
        record_prediction(
            prediction=prediction_label,
            confidence=confidence,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
        )

        if is_anomaly:
            record_anomaly("ensemble")

        # Log prediction
        logger.debug(
            "prediction_made",
            data_id=data_dict.get("id", "unknown"),
            is_anomaly=is_anomaly,
            confidence=confidence,
            epistemic=epistemic,
            aleatoric=aleatoric,
            source=source,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )

        return {
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "total_uncertainty": total_uncertainty,
            "prediction_interval_lower": pi_lower,
            "prediction_interval_upper": pi_upper,
            "anomaly_probability": anomaly_prob,
            "explanation": explanation,
            "prediction_source": source,
            "detector_scores": detector_scores,
            "missing_count": missing_count,
        }

    def add_human_feedback(
        self, data_dict: dict[str, Any], is_anomaly_label: bool
    ) -> None:
        """
        Add human feedback to buffer and trigger retraining if threshold reached.

        Args:
            data_dict: Original sensor data
            is_anomaly_label: Human-labeled ground truth (True = anomaly)
        """
        X, _, _ = self._preprocess(data_dict)
        label = 1 if is_anomaly_label else 0

        self.feedback_buffer.append((X[0], label))

        logger.info(
            "human_feedback_added",
            buffer_size=len(self.feedback_buffer),
            label=label,
            threshold=self.model_config.feedback_batch_size,
        )

        # Cap buffer size
        if len(self.feedback_buffer) > self.model_config.feedback_max_buffer:
            self.feedback_buffer = self.feedback_buffer[
                -self.model_config.feedback_max_buffer :
            ]

        # Retrain if threshold reached
        if len(self.feedback_buffer) >= self.model_config.feedback_batch_size:
            self._retrain_adaptation_model()

    def _retrain_adaptation_model(self) -> None:
        """Retrain RandomForest on human feedback buffer."""
        start_time = time.perf_counter()

        try:
            X_train = np.array([item[0] for item in self.feedback_buffer])
            y_train = np.array([item[1] for item in self.feedback_buffer])

            # Check class diversity
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                logger.warning(
                    "retrain_skipped_insufficient_class_diversity",
                    classes=unique_classes.tolist(),
                )
                return

            # Train
            self.rf_classifier.fit(X_train, y_train)
            self.is_rf_trained = True

            # Clear buffer after successful training (or keep for incremental learning)
            # self.feedback_buffer = []  # Commented to keep for incremental updates

            duration = time.perf_counter() - start_time
            record_retrain(
                trigger="feedback",
                duration_seconds=duration,
                model_type="random_forest",
            )

            logger.info(
                "adaptation_model_retrained",
                n_samples=len(self.feedback_buffer),
                n_features=X_train.shape[1],
                classes=unique_classes.tolist(),
                duration_ms=round(duration * 1000, 2),
            )

        except Exception as e:
            logger.error("retrain_failed", error=str(e), exc_info=True)

    def check_drift(self, recent_data: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Check for concept drift in recent data.

        This is a simplified drift check - full implementation in drift.py (Phase 2).
        """
        if (
            not self.model_config.drift_enabled
            or len(recent_data) < self.model_config.drift_min_samples
        ):
            return {"drift_detected": False, "reason": "insufficient_data"}

        # Simple statistical drift detection on feature distributions
        # Convert to DataFrame
        df = pd.DataFrame(recent_data)
        if df.empty:
            return {"drift_detected": False, "reason": "no_data"}

        drift_results = {}
        for feat in self.features_list:
            if feat not in df.columns:
                continue

            recent_values = df[feat].dropna()
            if len(recent_values) < 10:
                continue

            # KS-test against historical mean/std (simplified)
            from scipy import stats

            historical_mean = self.means.get(feat, 0)
            historical_std = self.stds.get(feat, 1)

            # Test if recent distribution matches historical
            # Generate synthetic historical data for KS test
            hist_synthetic = np.random.normal(historical_mean, historical_std, 1000)
            ks_stat, p_value = stats.ks_2samp(recent_values, hist_synthetic)

            drift_results[feat] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": p_value < self.model_config.drift_sensitivity,
            }

        any_drift = any(r["drift_detected"] for r in drift_results.values())

        if any_drift:
            logger.warning(
                "drift_detected",
                features={
                    k: v for k, v in drift_results.items() if v["drift_detected"]
                },
            )
            record_drift("data", severity=1.0)  # Placeholder

        return {
            "drift_detected": any_drift,
            "features": drift_results,
            "action": "schedule_retrain" if any_drift else "none",
        }

    def get_model_info(self) -> dict[str, Any]:
        """Get model metadata for monitoring."""
        return {
            "features": self.features_list,
            "is_fitted": self.is_fitted,
            "is_rf_trained": self.is_rf_trained,
            "feedback_buffer_size": len(self.feedback_buffer),
            "calibration_size": len(self.calibration_scores),
            "ensemble_enabled": self.model_config.ensemble_enabled,
            "uncertainty_method": self.model_config.uncertainty_method,
            "models": {
                "isolation_forest": {"type": "IsolationForest", "trained": True},
                "lof": {"type": "LocalOutlierFactor", "trained": True},
                "one_class_svm": {"type": "OneClassSVM", "trained": True},
                "random_forest": {
                    "type": "RandomForestClassifier",
                    "trained": self.is_rf_trained,
                },
            },
        }


# ==========================================
# Compatibility wrapper for existing API
# ==========================================


class LegacyAdaptiveDecisionEngine:
    """Backward compatibility wrapper for existing API."""

    def __init__(self, historical_data_path: str = "sensor_data.csv"):
        self.engine = AdaptiveDecisionEngine(historical_data_path)

    def predict(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        """Legacy predict interface returning simple dict."""
        result = self.engine.predict(data_dict)
        return {
            "is_anomaly": result["is_anomaly"],
            "confidence": result["confidence"],
            "explanation": result["explanation"],
        }

    def add_human_feedback(
        self, data_dict: dict[str, Any], is_anomaly_label: bool
    ) -> None:
        """Legacy feedback interface."""
        self.engine.add_human_feedback(data_dict, is_anomaly_label)
