"""Project Risk Prediction Model for APEX

This module implements ML-based project risk prediction using ensemble methods,
historical project data, and real-time metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProjectMetrics:
    """Container for project metrics"""
    planned_duration: float  # days
    actual_duration: float  # days
    budget: float
    spent: float
    team_size: int
    tasks_completed: int
    tasks_total: int
    critical_path_length: float
    resource_utilization: float  # 0-1
    velocity: float  # story points per sprint
    burndown_variance: float
    dependencies_count: int
    blocked_tasks: int
    
    def to_dict(self) -> Dict:
        return self.__dict__


class RiskPredictor:
    """Predicts project delivery risk using ensemble ML models"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.scaler = StandardScaler()
        self.schedule_risk_model = None
        self.budget_risk_model = None
        self.quality_risk_model = None
        
        if model_path:
            self.load_models(model_path)
        else:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize fresh ML models"""
        # Schedule delay risk (binary classification)
        self.schedule_risk_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42
        )
        
        # Budget overrun prediction (regression)
        self.budget_risk_model = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=10,
            random_state=42
        )
        
        # Quality risk score (regression 0-100)
        self.quality_risk_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
    
    def extract_features(self, metrics: ProjectMetrics) -> np.ndarray:
        """Extract ML features from project metrics"""
        features = [
            # Progress indicators
            metrics.tasks_completed / max(metrics.tasks_total, 1),
            metrics.actual_duration / max(metrics.planned_duration, 1),
            metrics.spent / max(metrics.budget, 1),
            
            # Team metrics
            metrics.team_size,
            metrics.resource_utilization,
            metrics.velocity,
            
            # Risk indicators
            metrics.burndown_variance,
            metrics.blocked_tasks / max(metrics.tasks_total, 1),
            metrics.dependencies_count,
            metrics.critical_path_length,
            
            # Derived features
            metrics.spent / max(metrics.actual_duration, 1),  # Burn rate
            (metrics.budget - metrics.spent) / max(metrics.budget, 1),  # Budget remaining %
            (metrics.tasks_total - metrics.tasks_completed) / max(metrics.velocity, 1)  # Sprints remaining
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Train models on historical project data
        
        Args:
            historical_data: DataFrame with columns matching ProjectMetrics
                           plus outcome columns: 'schedule_delayed', 'budget_overrun_pct', 'quality_score'
        
        Returns:
            Dictionary of model scores
        """
        logger.info(f"Training on {len(historical_data)} historical projects")
        
        # Extract features
        X = np.vstack([
            self.extract_features(ProjectMetrics(**row.to_dict()))
            for _, row in historical_data.iterrows()
        ])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train schedule risk model
        y_schedule = historical_data['schedule_delayed'].values
        self.schedule_risk_model.fit(X_scaled, y_schedule)
        schedule_score = self.schedule_risk_model.score(X_scaled, y_schedule)
        
        # Train budget risk model
        y_budget = historical_data['budget_overrun_pct'].values
        self.budget_risk_model.fit(X_scaled, y_budget)
        budget_score = self.budget_risk_model.score(X_scaled, y_budget)
        
        # Train quality risk model
        y_quality = historical_data['quality_score'].values
        self.quality_risk_model.fit(X_scaled, y_quality)
        quality_score = self.quality_risk_model.score(X_scaled, y_quality)
        
        scores = {
            'schedule_accuracy': schedule_score,
            'budget_accuracy': budget_score,
            'quality_accuracy': quality_score
        }
        
        logger.info(f"Training complete. Scores: {scores}")
        return scores
    
    def predict_risk(self, metrics: ProjectMetrics) -> Dict[str, float]:
        """Predict comprehensive project risk
        
        Returns:
            Dictionary with risk predictions:
            - schedule_delay_probability: 0-1
            - budget_overrun_estimate: percentage
            - quality_risk_score: 0-100
            - overall_risk_level: LOW/MEDIUM/HIGH/CRITICAL
        """
        # Extract and scale features
        X = self.extract_features(metrics)
        X_scaled = self.scaler.transform(X)
        
        # Predict schedule delay probability
        schedule_proba = self.schedule_risk_model.predict_proba(X_scaled)[0][1]
        
        # Predict budget overrun
        budget_overrun = max(0, self.budget_risk_model.predict(X_scaled)[0])
        
        # Predict quality risk
        quality_risk = max(0, min(100, self.quality_risk_model.predict(X_scaled)[0]))
        
        # Calculate overall risk level
        risk_score = (
            schedule_proba * 0.4 +
            min(budget_overrun / 50, 1.0) * 0.35 +
            quality_risk / 100 * 0.25
        )
        
        if risk_score < 0.25:
            risk_level = "LOW"
        elif risk_score < 0.5:
            risk_level = "MEDIUM"
        elif risk_score < 0.75:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return {
            'schedule_delay_probability': round(schedule_proba, 3),
            'budget_overrun_estimate_pct': round(budget_overrun, 2),
            'quality_risk_score': round(quality_risk, 1),
            'overall_risk_score': round(risk_score, 3),
            'overall_risk_level': risk_level,
            'confidence_intervals': self._calculate_confidence(X_scaled)
        }
    
    def _calculate_confidence(self, X: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate prediction confidence intervals"""
        # Use tree-based variance for confidence
        schedule_trees = [
            tree.predict_proba(X)[0][1]
            for tree in self.schedule_risk_model.estimators_
        ]
        
        return {
            'schedule_ci': (
                round(np.percentile(schedule_trees, 5), 3),
                round(np.percentile(schedule_trees, 95), 3)
            )
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from schedule risk model"""
        feature_names = [
            'completion_pct', 'duration_ratio', 'budget_ratio',
            'team_size', 'resource_util', 'velocity',
            'burndown_var', 'blocked_ratio', 'dependencies',
            'critical_path', 'burn_rate', 'budget_remaining',
            'sprints_remaining'
        ]
        
        importances = self.schedule_risk_model.feature_importances_
        return dict(zip(feature_names, importances))
    
    def save_models(self, path: str):
        """Save trained models to disk"""
        joblib.dump({
            'scaler': self.scaler,
            'schedule_model': self.schedule_risk_model,
            'budget_model': self.budget_risk_model,
            'quality_model': self.quality_risk_model
        }, path)
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load trained models from disk"""
        models = joblib.load(path)
        self.scaler = models['scaler']
        self.schedule_risk_model = models['schedule_model']
        self.budget_risk_model = models['budget_model']
        self.quality_risk_model = models['quality_model']
        logger.info(f"Models loaded from {path}")


# Example usage
if __name__ == "__main__":
    # Create sample project metrics
    current_project = ProjectMetrics(
        planned_duration=90,
        actual_duration=45,
        budget=100000,
        spent=55000,
        team_size=8,
        tasks_completed=45,
        tasks_total=100,
        critical_path_length=30,
        resource_utilization=0.85,
        velocity=15,
        burndown_variance=0.15,
        dependencies_count=25,
        blocked_tasks=3
    )
    
    # Initialize predictor (would normally train first)
    predictor = RiskPredictor()
    
    # Predict risk
    risk_assessment = predictor.predict_risk(current_project)
    
    print("Project Risk Assessment:")
    for key, value in risk_assessment.items():
        print(f"  {key}: {value}")
