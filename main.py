import numpy as np
import pandas as pd
import gradio as gr
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Dict, Optional
from pathlib import Path
import logging


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
RULE_MODEL_PATH = MODEL_DIR / "rule_model.pkl"
ML_MODEL_PATH = MODEL_DIR / "ml_model.pkl"
MODEL_NAMES = {
    "rule": "Wait! I am machine learning model",
    "ml": "No No! I am machine learning model",
}
DEFAULT_INPUTS = {
    "sqmt": 140,
    "bedrooms": 3,
    "distance": 8
}
PRICE_MIN = 400
PRICE_MAX = 3000
SQMT_MIN = 15
SQMT_MAX = 200
BEDROOMS_MIN = 1
BEDROOMS_MAX = 6
DISTANCE_MIN = 0.1
DISTANCE_MAX = 50
NOISE_MIN = -50
NOISE_MAX = 50


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Custom rule-based model
class RuleBasedRegressor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        sqmt = X.iloc[:, 0].values if isinstance(X, pd.DataFrame) else X[:, 0]
        bedrooms = X.iloc[:, 1].values if isinstance(X, pd.DataFrame) else X[:, 1]
        distance = X.iloc[:, 2].values if isinstance(X, pd.DataFrame) else X[:, 2]
        return price_func(sqmt, bedrooms, distance)

def price_func(sqmt: np.ndarray, bedrooms: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Calculate house price with logical constraints."""
    distance = np.clip(distance, DISTANCE_MIN, DISTANCE_MAX)
    base_price = PRICE_MIN + 0.15 * sqmt + 200 * bedrooms - 5.5 * distance
    large_home_bonus = (sqmt > 140) * (bedrooms > 3) * 400
    small_home_penalty = (sqmt < 30) * (bedrooms < 2) * (-100)
    return np.clip(base_price + large_home_bonus - small_home_penalty, PRICE_MIN, PRICE_MAX)

def confidence_func(train_x: np.ndarray, train_y: np.ndarray, x: np.ndarray, y: float) -> float:
    """Calculate prediction confidence based on proximity to training data."""
    distances = np.linalg.norm(train_x - x, axis=1)
    closest_indices = np.argsort(distances)[:5]
    closest_y = train_y[closest_indices]
    y_value = y if isinstance(y, (int, float)) else y[0]
    
    distance_to_pred = np.abs(closest_y - y_value)
    relative_diff = distance_to_pred / (np.abs(y_value) + 1e-10)
    closeness = 1 - relative_diff
    logits = np.where(closeness >= 0.6, 8 * closeness, -8 * (1 - closeness))
    exp_logits = np.exp(logits - np.max(logits))
    softmax_scores = exp_logits / np.sum(exp_logits)
    return round(np.max(softmax_scores), 3)

def generate_synthetic_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic housing data."""
    rng = np.random.default_rng(seed)
    sqmt = np.clip(rng.normal(140, 40, n_samples), SQMT_MIN, SQMT_MAX)
    bedrooms = np.clip(rng.poisson(4, n_samples), BEDROOMS_MIN, BEDROOMS_MAX).astype(int)
    distance = np.clip(rng.gamma(3, 5, n_samples), DISTANCE_MIN, DISTANCE_MAX)
    price = np.clip(price_func(sqmt, bedrooms, distance) + rng.uniform(NOISE_MIN, NOISE_MAX, n_samples), 
                    PRICE_MIN, PRICE_MAX)
    return pd.DataFrame({"sqmt": sqmt, "bedrooms": bedrooms, "distance": distance, "price": price})

def calculate_rule_feature_importance() -> Dict[str, float]:
    """Calculate feature importance for the rule-based model based on coefficients and ranges."""
    coef_sqmt = 0.15
    coef_bedrooms = 200
    coef_distance = -5.5

    impact_sqmt = coef_sqmt * (SQMT_MAX - SQMT_MIN)
    impact_bedrooms = coef_bedrooms * (BEDROOMS_MAX - BEDROOMS_MIN)
    impact_distance = abs(coef_distance) * (DISTANCE_MAX - DISTANCE_MIN)
    total_impact = impact_sqmt + impact_bedrooms + impact_distance

    # Normalize to get importance (as fractions summing to 1)
    importance = {
        "sqmt": impact_sqmt / total_impact,
        "bedrooms": impact_bedrooms / total_impact,
        "distance": impact_distance / total_impact
    }
    return importance

def train_model(data: pd.DataFrame, model_type: str) -> Dict:
    """Train a model and save it."""
    X = data[["sqmt", "bedrooms", "distance"]]
    preprocessor = ColumnTransformer([("num", StandardScaler(), ["sqmt", "bedrooms", "distance"])])
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)

    if model_type == "rule":
        y = price_func(X["sqmt"], X["bedrooms"], X["distance"])
        regressor = RuleBasedRegressor()
        model_path = RULE_MODEL_PATH
        feature_importances = calculate_rule_feature_importance()
    else:  # ml
        y = data["price"]
        regressor = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5, min_samples_split=5, random_state=42
        )
        model_path = ML_MODEL_PATH
        regressor.fit(X_transformed, y)
        feature_importances = dict(zip(["sqmt", "bedrooms", "distance"], regressor.feature_importances_))

    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
    y_scaler = StandardScaler()
    y_transformed = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    y_pipeline = Pipeline([("scaler", y_scaler)])

    reference_data = {
        "pipeline": pipeline,
        "y_pipeline": y_pipeline,
        "X_transformed": X_transformed,
        "y_transformed": y_transformed,
        "raw_y": y,
        "params": {"Feature Importances": feature_importances},
    }
    joblib.dump(reference_data, model_path)
    logger.info(f"{model_type.capitalize()} model trained and saved to {model_path}")
    return reference_data

def predict_model(sqmt: float, bedrooms: int, distance: float, model_type: str) -> Tuple[str, str]:
    """Make a prediction with confidence."""
    try:
        # Clip inputs to valid ranges
        sqmt = np.clip(sqmt, SQMT_MIN, SQMT_MAX)
        bedrooms = np.clip(bedrooms, BEDROOMS_MIN, BEDROOMS_MAX)
        distance = np.clip(distance, DISTANCE_MIN, DISTANCE_MAX)

        reference_data = joblib.load(RULE_MODEL_PATH) if model_type == "rule" else joblib.load(ML_MODEL_PATH)
        pipeline = reference_data["pipeline"]
        y_pipeline = reference_data["y_pipeline"]
        X_transformed = reference_data["X_transformed"]
        y_transformed = reference_data["y_transformed"]

        input_data = pd.DataFrame([[sqmt, bedrooms, distance]], columns=["sqmt", "bedrooms", "distance"])
        if model_type == "rule":
            predicted_price = price_func(sqmt, bedrooms, distance)
        else:
            predicted_price = pipeline.predict(input_data)[0]
        input_transformed = pipeline.named_steps["preprocessor"].transform(input_data)
        predicted_price_transformed = y_pipeline.transform([[predicted_price]])[0][0]
        confidence = confidence_func(X_transformed, y_transformed, input_transformed, predicted_price_transformed)
        return f"€{predicted_price:.2f}", f"{confidence:.3f}"
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Error", "N/A"

def create_gradio_interface(rule_data, synthetic_data) -> gr.Blocks:
    """Create a polished Gradio interface."""

    with gr.Blocks(title="House Price Predictor") as demo:
        gr.Markdown(
            """
            # 🏠 House Price Predictor
            Compare predictions from two different models with confidence scores.
            """
        )

        with gr.Row():
            # Rule-Based Model
            with gr.Column():
                gr.Markdown(f"## {MODEL_NAMES['rule']}")
                with gr.Group():
                    sqft1 = gr.Number(
                        label="Square meters",
                        value=DEFAULT_INPUTS["sqmt"],
                        minimum=0,
                        maximum=5000,
                        step=5
                    )
                    bedrooms1 = gr.Slider(
                        label="Bedrooms",
                        value=DEFAULT_INPUTS["bedrooms"],
                        minimum=0,
                        maximum=20,
                        step=1
                    )
                    distance1 = gr.Slider(
                        label="Distance from city center (km)",
                        value=DEFAULT_INPUTS["distance"],
                        minimum=DISTANCE_MIN,
                        maximum=100,
                        step=0.5
                    )

                with gr.Row():
                    btn1 = gr.Button("Predict", variant="primary")
                
                with gr.Accordion("Training Data", open=False):
                    data_with_predictions = rule_data.copy()
                    data_with_predictions["predicted_price"] = data_with_predictions.apply(
                        lambda row: predict_model(row["sqmt"], row["bedrooms"], row["distance"], model_type="rule")[0], axis=1
                    )
                    gr.DataFrame(data_with_predictions.round(2))

                with gr.Group():
                    price_output1 = gr.Textbox(label="Price")
                    confidence_output1 = gr.Textbox(label="Confidence")

                with gr.Accordion("Feature Importances", open=False):
                    rule_data = joblib.load(RULE_MODEL_PATH)
                    for feature, imp in rule_data["params"]["Feature Importances"].items():
                        gr.Markdown(f"- **{feature.capitalize()}**: {imp:.1%}")

            # ML Model
            with gr.Column():
                gr.Markdown(f"## {MODEL_NAMES['ml']}")
                with gr.Group():
                    sqft2 = gr.Number(
                        label="Square meters",
                        value=DEFAULT_INPUTS["sqmt"],
                        minimum=0,
                        maximum=5000,
                        step=5
                    )
                    bedrooms2 = gr.Slider(
                        label="Bedrooms",
                        value=DEFAULT_INPUTS["bedrooms"],
                        minimum=0,
                        maximum=20,
                        step=1
                    )
                    distance2 = gr.Slider(
                        label="Distance from city center (km)",
                        value=DEFAULT_INPUTS["distance"],
                        minimum=DISTANCE_MIN,
                        maximum=100,
                        step=0.5
                    )

                with gr.Row():
                    btn2 = gr.Button("Predict", variant="primary")

                with gr.Accordion("Training Data", open=False):
                    data_with_predictions_2 = synthetic_data.copy()
                    data_with_predictions_2.drop(columns=["price"], inplace=True)
                    data_with_predictions_2["predicted_price"] = data_with_predictions_2.apply(
                        lambda row: predict_model(row["sqmt"], row["bedrooms"], row["distance"], model_type="ml")[0], axis=1
                    )
                    gr.DataFrame(data_with_predictions_2.round(2))

                with gr.Group():
                    price_output2 = gr.Textbox(label="Price")
                    confidence_output2 = gr.Textbox(label="Confidence")

                with gr.Accordion("Feature Importances", open=False):
                    ml_data = joblib.load(ML_MODEL_PATH)
                    for feature, imp in ml_data["params"]["Feature Importances"].items():
                        gr.Markdown(f"- **{feature.capitalize()}**: {imp:.1%}")

        btn1.click(
            fn=lambda s, b, d: predict_model(s, b, d, "rule"),
            inputs=[sqft1, bedrooms1, distance1],
            outputs=[price_output1, confidence_output1]
        )
        btn2.click(
            fn=lambda s, b, d: predict_model(s, b, d, "ml"),
            inputs=[sqft2, bedrooms2, distance2],
            outputs=[price_output2, confidence_output2]
        )

        gr.Markdown(
            """
            ---
            Copyright © 2025 [Binay Kumar Pradhan](https://binaypradhan.com)
            """
        )

    return demo

def main():
    """Main execution function."""
    # Clean up old model files
    for path in [RULE_MODEL_PATH, ML_MODEL_PATH]:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Removed existing model file: {path}")

    # Generate and train models
    synthetic_data = generate_synthetic_data()
    rule_data = pd.DataFrame({
        "sqmt": np.linspace(SQMT_MIN + 13, SQMT_MAX, 20),
        "bedrooms": np.random.default_rng(42).choice(range(BEDROOMS_MIN, BEDROOMS_MAX + 1), 20),
        "distance": np.linspace(DISTANCE_MIN, DISTANCE_MAX, 20),
    })
    train_model(rule_data, "rule")
    train_model(synthetic_data, "ml")

    # Launch interface
    interface = create_gradio_interface(rule_data, synthetic_data.sample(20))
    interface.launch()

if __name__ == "__main__":
    main()
