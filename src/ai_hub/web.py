import streamlit as st
import requests
import pandas as pd
import json
from typing import List, Dict, Any

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AI Hub", page_icon="🤖", layout="wide")


def check_api():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE}/", timeout=2)
        return response.status_code == 200
    except:
        return False


def train_model(input_dim: int, hidden_dim: int, output_dim: int, epochs: int):
    """Train a model via API."""
    response = requests.post(
        f"{API_BASE}/train",
        json={
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "epochs": epochs,
        },
    )
    return response.json()


def predict(data: List[List[float]]):
    """Make predictions via API."""
    response = requests.post(f"{API_BASE}/predict", json={"data": data})
    return response.json()


def get_models():
    """Get list of models."""
    response = requests.get(f"{API_BASE}/models")
    return response.json()


def save_model(name: str):
    """Save model."""
    response = requests.post(f"{API_BASE}/save", json={"model_name": name})
    return response.json()


def load_model(name: str):
    """Load model."""
    response = requests.post(f"{API_BASE}/load/{name}")
    return response.json()


def delete_model(name: str):
    """Delete model."""
    response = requests.delete(f"{API_BASE}/models/{name}")
    return response.json()


def get_status():
    """Get API status."""
    response = requests.get(f"{API_BASE}/status")
    return response.json()


def main():
    st.title("🤖 AI Hub - Web Interface")

    api_running = check_api()

    if not api_running:
        st.error("⚠️ API is not running. Please start it with: `ai-hub api start`")
        st.code("ai-hub api start", language="bash")
        return

    st.success("✅ API Connected")

    tab1, tab2, tab3 = st.tabs(["🎯 Train", "🔮 Predict", "💾 Models"])

    with tab1:
        st.header("Train a Model")

        col1, col2 = st.columns(2)

        with col1:
            input_dim = st.number_input("Input Dimension", min_value=1, value=128)
            hidden_dim = st.number_input("Hidden Dimension", min_value=1, value=256)

        with col2:
            output_dim = st.number_input("Output Dimension", min_value=1, value=10)
            epochs = st.number_input("Epochs", min_value=1, value=10)

        if st.button("Train Model", type="primary"):
            with st.spinner("Training..."):
                result = train_model(input_dim, hidden_dim, output_dim, epochs)

                if result.get("status") == "trained":
                    st.success("Training complete! 🎉")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Final Loss", f"{result['training']['final_loss']:.4f}")
                    col2.metric("Accuracy", f"{result['evaluation']['accuracy']:.2%}")
                    col3.metric("Epochs", result["training"]["epochs"])
                else:
                    st.error(f"Training failed: {result}")

    with tab2:
        st.header("Make Predictions")

        status = get_status()

        if not status.get("trained"):
            st.warning("No trained model. Please train a model first.")
        else:
            st.info(f"Model ready: {status.get('model_loaded', False)}")

            num_samples = st.number_input(
                "Number of samples", min_value=1, max_value=10, value=1
            )
            input_dim = st.number_input("Input dimension", min_value=1, value=128)

            data = []
            for i in range(num_samples):
                cols = st.columns(min(input_dim, 8))
                row = []
                for j in range(min(input_dim, 8)):
                    val = cols[j].number_input(
                        f"Sample {i + 1}, Feature {j + 1}", value=0.0, key=f"s{i}_f{j}"
                    )
                    row.append(val)
                if input_dim > 8:
                    row.extend([0.0] * (input_dim - 8))
                data.append(row)

            if st.button("Predict", type="primary"):
                with st.spinner("Predicting..."):
                    result = predict(data)

                    if "predictions" in result:
                        st.success("Predictions complete! 🎉")

                        df = pd.DataFrame(
                            {
                                "Sample": range(len(result["predictions"])),
                                "Prediction": result["predictions"],
                                "Confidence": [
                                    f"{max(p):.2%}" for p in result["probabilities"]
                                ],
                            }
                        )
                        st.table(df)
                    else:
                        st.error(f"Prediction failed: {result}")

    with tab3:
        st.header("Model Management")

        if st.button("🔄 Refresh Models"):
            st.rerun()

        result = get_models()

        if result["count"] > 0:
            models_data = []
            for model in result["models"]:
                md = model.get("metadata", {})
                models_data.append(
                    {
                        "Name": model["name"],
                        "Input Dim": md.get("input_dim", "N/A"),
                        "Hidden Dim": md.get("hidden_dim", "N/A"),
                        "Output Dim": md.get("output_dim", "N/A"),
                    }
                )

            df = pd.DataFrame(models_data)
            st.table(df)

            col1, col2 = st.columns(2)

            with col1:
                model_name = st.text_input("Model name to save")
                if st.button("💾 Save Current Model"):
                    if model_name:
                        result = save_model(model_name)
                        st.success(f"Model saved: {result.get('model_path')}")
                    else:
                        st.error("Please enter a model name")

            with col2:
                load_name = st.selectbox(
                    "Model to load", [m["name"] for m in result["models"]]
                )
                if st.button("📂 Load Model"):
                    result = load_model(load_name)
                    st.success(f"Model loaded: {result.get('model_name')}")

            delete_name = st.selectbox(
                "Model to delete", [m["name"] for m in result["models"]]
            )
            if st.button("🗑️ Delete Model", type="primary"):
                result = delete_model(delete_name)
                st.success(f"Model deleted: {result.get('status')}")
                st.rerun()
        else:
            st.info("No saved models. Train and save a model to see it here.")

    st.divider()
    st.caption("AI Hub - Unified ML Platform | Use `ai-hub --help` for CLI commands")


if __name__ == "__main__":
    main()
