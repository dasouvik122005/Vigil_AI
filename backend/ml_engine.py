from sklearn.ensemble import IsolationForest, RandomForestClassifier
import pandas as pd
import numpy as np

class AdaptiveDecisionEngine:
    def __init__(self, historical_data_path="sensor_data.csv"):
        # 1. Unsupervised Anomaly Detection (Base Model)
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        
        # 2. Supervised Adaptation Model (Learns from humans)
        self.rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # State
        self.is_rf_trained = False
        self.feedback_buffer = []  # Stores (features_dict, human_label)
        self.features_list = ['temperature', 'pressure', 'vibration']
        
        # Load and train initial base model
        try:
            df = pd.read_csv(historical_data_path)
            # Impute missing values with mean just for training the baseline
            self.means = df[self.features_list].mean().to_dict()
            self.stds = df[self.features_list].std().to_dict()
            
            X_train = df[self.features_list].fillna(df[self.features_list].mean())
            self.iso_forest.fit(X_train)
            print("Successfully trained IsolationForest on historical data.")
        except Exception as e:
            print(f"Error training base model: {e}")
            self.means = {'temperature': 25, 'pressure': 1.2, 'vibration': 0.3}
            self.stds = {'temperature': 3, 'pressure': 0.1, 'vibration': 0.05}

    def _preprocess(self, data_dict):
        """Imputes missing data using historical means (required for sklearn models)"""
        features = []
        missing_count = 0
        for f in self.features_list:
            if data_dict.get(f) is None:
                features.append(self.means[f])
                missing_count += 1
            else:
                features.append(data_dict[f])
        return np.array([features]), missing_count

    def predict(self, data_dict):
        X, missing_count = self._preprocess(data_dict)
        
        # Base confidence calculation based on data completeness
        # If 1 feature is missing, max confidence is ~66%. If 2 missing, max is ~33%.
        data_completeness_ratio = (len(self.features_list) - missing_count) / len(self.features_list)
        
        # Step 1: Base Prediction using unsupervised Isolation Forest
        # Returns -1 for anomaly, 1 for normal
        iso_pred = self.iso_forest.predict(X)[0]
        iso_score = self.iso_forest.decision_function(X)[0] # negative for anomalies, positive for normal
        
        is_anomaly = iso_pred == -1
        
        # Step 2: Adaptive Override (if human model is trained)
        prediction_source = "IsolationForest (Unsupervised Base)"
        if self.is_rf_trained:
            # rf predicts 1 for anomaly, 0 for normal (based on how we label feedback)
            rf_pred = self.rf_classifier.predict(X)[0]
            rf_proba = self.rf_classifier.predict_proba(X)[0]
            
            # If the supervised model is highly confident, we override the base model
            max_proba = max(rf_proba)
            if max_proba > 0.7: 
                is_anomaly = bool(rf_pred == 1)
                prediction_source = "RandomForest (Human-Adapted)"
                # Boost confidence slightly because it's human-guided
                data_completeness_ratio = min(1.0, data_completeness_ratio + 0.1)

        # Step 3: Confidence Scoring
        # Combine data completeness with model decision boundary distance
        normalized_score = min(1.0, abs(iso_score) * 2) # closer to 0 means uncertain
        confidence = float(data_completeness_ratio * (0.5 + (0.5 * normalized_score)))
        
        # Step 4: Explainable AI (XAI)
        explanation = f"Prediction made by {prediction_source}. "
        if missing_count > 0:
            explanation += f"WARNING: {missing_count} sensors offline. "
            
        if is_anomaly:
            explanation += "Detected deviations: "
            deviations = []
            for i, f in enumerate(self.features_list):
                if data_dict.get(f) is not None:
                    z_score = abs(data_dict[f] - self.means[f]) / self.stds[f]
                    if z_score > 2.0:
                        deviations.append(f"{f} is {z_score:.1f} standard deviations from mean")
            
            if deviations:
                explanation += ", ".join(deviations) + "."
            else:
                explanation += "Complex multivariate anomaly."
        else:
            explanation += "All available sensor readings are within normal historical boundaries."

        return {
            "is_anomaly": is_anomaly,
            "confidence": min(0.99, max(0.01, confidence)),
            "explanation": explanation
        }

    def add_human_feedback(self, data_dict, is_anomaly_label):
        """
        Takes human corrections and adds them to the buffer.
        If buffer is large enough, retrains the adaptation model.
        """
        # We store 1 for anomaly, 0 for normal
        label = 1 if is_anomaly_label else 0
        X, _ = self._preprocess(data_dict)
        self.feedback_buffer.append((X[0], label))
        
        print(f"Added human feedback. Buffer size: {len(self.feedback_buffer)}/5")
        
        # Retrain every 5 human interventions
        if len(self.feedback_buffer) >= 5:
            self._retrain_adaptation_model()

    def _retrain_adaptation_model(self):
        print("Retraining adaptation model on human feedback...")
        X_train = np.array([item[0] for item in self.feedback_buffer])
        y_train = np.array([item[1] for item in self.feedback_buffer])
        
        # Only train if we have both classes (otherwise RandomForest crashes)
        if len(set(y_train)) > 1:
            self.rf_classifier.fit(X_train, y_train)
            self.is_rf_trained = True
            print("Adaptation model successfully updated.")
        else:
            print("Not enough class diversity to retrain yet.")
