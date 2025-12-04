import os
import numbers
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import shap # For the Explanation object type hint
import numpy as np

# Load environment variables from .env file
load_dotenv()

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
AI_COMMENTARY_FEATURE_COUNT = 10 # Number of features to send to the LLM

# --- Helper functions adapted from your shap_app.py ---

def _format_feature_value(value):
    """Formats feature values for the prompt."""
    if isinstance(value, numbers.Number):
        # Check if it's an integer or float
        if isinstance(value, float) and not value.is_integer():
             return f"{float(value):.4f}"
        return str(value)
    return str(value)

def _summarize_top_features(top_features_df):
    """Converts the top features DataFrame into a numbered list string."""
    lines = []
    for idx, (_, row) in enumerate(top_features_df.iterrows(), start=1):
        feature_name = str(row["Feature"])
        feature_value = _format_feature_value(row["Value"])
        shap_value = _format_feature_value(row["SHAP Value"])
        lines.append(f"{idx}. {feature_name} (value={feature_value}, shap={shap_value})")
    return "\n".join(lines)

@st.cache_resource
def _get_openai_client():
    """
    Initializes and caches the OpenAI client.
    Reads the API key from the .env file.
    """
    # Try to load again, just in case
    load_dotenv() 
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # --- NEW, more helpful error ---
        cwd = os.getcwd()
        env_path = os.path.join(cwd, '.env')
        file_exists = os.path.exists(env_path)
        
        error_message = (
            "**OpenAI API key missing.** `os.getenv('OPENAI_API_KEY')` returned `None`.\n\n"
            "**Please check the following:**\n\n"
            f"1. **File Location:** Is your `.env` file in this exact folder?\n   `{env_path}`\n   (File exists: **{file_exists}**)\n\n"
            "2. **File Format:** Does your `.env` file contain a line *without quotes*?\n   **Correct:** `OPENAI_API_KEY=sk-yourkey...`\n   **Incorrect:** `OPENAI_API_KEY=\"sk-yourkey...\"`\n\n"
            "3. **Restart App:** Have you **restarted** your Streamlit app (e.g., `streamlit run streamlit_app.py`) after creating/editing the `.env` file?"
        )
        return None, error_message
        # --- END NEW ERROR ---
    try:
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as exc:
        return None, f"Unable to initialize OpenAI client: {exc}"

def _generate_ai_commentary(top_features_df, prediction_class, prediction_confidence, actual_label_text):
    """
    Sends the prompt to the OpenAI API and returns the text response.
    """
    if top_features_df.empty:
        return None, "No feature contributions available for commentary."
    
    client, client_error = _get_openai_client()
    if client is None:
        return None, client_error
        
    feature_summary = _summarize_top_features(top_features_df)
    
    # Determine predicted class text
    pred_class_text = f"Class {prediction_class} (Confidence: {prediction_confidence:.2%})"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a machine learning analyst who explains SHAP feature contributions "
                "to business stakeholders in friendly, precise language. Focus on *why* the prediction was made."
            ),
        },
        {
            "role": "user",
            "content": (
                "Model prediction summary:\n"
                f"- Model Prediction: {pred_class_text}\n"
                f"- Actual Value: {actual_label_text}\n\n"
                f"Top {len(top_features_df)} contributing features (highest impact first):\n"
                f"{feature_summary}\n\n"
                "Write two short paragraphs (under 160 words total).\n"
                "Paragraph 1: Explain *why* the model made this prediction. How do the top features (and their values) push the prediction toward the final probability? "
                "Paragraph 2: Compare the prediction to the actual value. If they are different, briefly suggest why the model might have been mistaken based on the feature values. If they are the same, confirm that the features provide a strong explanation."
            ),
        },
    ]
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=220,
        )
        content = response.choices[0].message.content.strip()
        return content, None
    except Exception as exc:
        return None, f"OpenAI request failed: {exc}"

# --- Main public function to be called by streamlit_app.py ---

def get_llm_explanation(explanation: shap.Explanation, actual_target: any, prediction_prob_class_1: float):
    """
    Takes a SHAP Explanation object and generates a natural language explanation.
    
    Returns:
        (str, None) on success: The explanation text.
        (None, str) on failure: The error message.
    """
    
    try:
        # 1. Convert SHAP Explanation to the required DataFrame format
        contributions_df = pd.DataFrame({
            'Feature': explanation.feature_names,
            'Value': explanation.data,
            'SHAP Value': explanation.values
        })
        
        # 2. Sort by impact
        contributions_df['Absolute Impact'] = np.abs(contributions_df['SHAP Value'])
        contributions_df = contributions_df.sort_values(by='Absolute Impact', ascending=False).drop(columns=['Absolute Impact'])
        
        # 3. Get top features for the prompt
        top_features_for_ai = contributions_df.head(AI_COMMENTARY_FEATURE_COUNT)
        
        # 4. Determine prediction class and confidence
        prediction_class = 1 if prediction_prob_class_1 >= 0.5 else 0
        prediction_confidence = prediction_prob_class_1 if prediction_class == 1 else 1 - prediction_prob_class_1
        
        # 5. Format actual target
        actual_label_text = f"Class {actual_target}"

        # 6. Call the generator
        commentary, error = _generate_ai_commentary(
            top_features_for_ai,
            prediction_class=prediction_class,
            prediction_confidence=prediction_confidence,
            actual_label_text=actual_label_text
        )
        
        if error:
            return None, error
        else:
            return commentary, None

    except Exception as e:
        return None, f"An error occurred preparing data for the LLM: {e}"

