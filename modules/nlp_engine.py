"""
NLP Engine Module (Team 2 — NLP & Intent Recognition)
=======================================================
Handles Google Gemini API integration for natural language
understanding, intent classification, and entity extraction.
"""

import json
import re
import time
import google.generativeai as genai
from utils.config import GOOGLE_API_KEY, GEMINI_MODEL
from prompts.system_prompts import SYSTEM_PROMPT, INSIGHT_PROMPT


# List of models to try in order (fallback chain)
FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]


def initialize_gemini():
    """Initialize the Google Gemini API client."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here_replace_me":
        return None
    genai.configure(api_key=GOOGLE_API_KEY)

    # Try the configured model first, then fallbacks
    models_to_try = [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            # Quick test to verify the model works
            model.generate_content("test", generation_config={"max_output_tokens": 5})
            print(f"✅ Using Gemini model: {model_name}")
            return model
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"⚠️ Model {model_name} quota exceeded, trying next...")
                continue
            elif "not found" in str(e).lower():
                print(f"⚠️ Model {model_name} not available, trying next...")
                continue
            else:
                # Other error — model might still work for real queries
                model = genai.GenerativeModel(model_name)
                return model

    # If all fail, return the default model anyway
    return genai.GenerativeModel(GEMINI_MODEL)


def get_chat_session(model, dataset_summary):
    """Create a new chat session with system context."""
    system_prompt = SYSTEM_PROMPT.format(dataset_summary=dataset_summary)
    chat = model.start_chat(history=[
        {"role": "user", "parts": [f"System context: {system_prompt}"]},
        {"role": "model", "parts": ["Understood! I'm your AI Business Intelligence Assistant. I have access to the company's sales, HR, and finance data. How can I help you analyze your data today?"]}
    ])
    return chat


def send_message(chat, message, max_retries=3):
    """
    Send a message to Gemini and parse the response.
    Includes retry logic with exponential backoff for rate limit errors.
    Returns: (raw_response_text, parsed_json_or_None)
    """
    for attempt in range(max_retries):
        try:
            response = chat.send_message(message)
            response_text = response.text

            # Try to extract JSON from the response
            parsed_json = extract_json_from_response(response_text)

            return response_text, parsed_json

        except Exception as e:
            error_str = str(e)

            # Handle rate limit (429) errors with retry
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 15 * (2 ** attempt)  # 15s, 30s, 60s
                    print(f"⏳ Rate limited. Waiting {wait_time}s before retry ({attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = (
                        "⚠️ **API Quota Exceeded**\n\n"
                        "Your Gemini API free tier quota has been exhausted. Options:\n"
                        "1. **Wait** ~1 minute and try again\n"
                        "2. **Enable billing** in [Google AI Studio](https://aistudio.google.com/) for higher limits\n"
                        "3. The **Dashboard** and **Forecast** pages work without the API!\n\n"
                        "The chatbot will retry automatically — just ask your question again in a minute."
                    )
                    return error_msg, None
            else:
                error_msg = f"I encountered an error processing your request: {error_str}"
                return error_msg, None

    return "Unexpected error occurred. Please try again.", None


def extract_json_from_response(text):
    """Extract JSON object from Gemini's response text."""
    try:
        # Look for JSON block in markdown code fence
        json_match = re.search(r'```json\s*\n(.*?)\n\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            return json.loads(json_str)

        # Try to find raw JSON object
        json_match = re.search(r'\{[\s\S]*"intent"[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group(0))

    except json.JSONDecodeError:
        pass

    return None


def generate_insight(model, query, title, results_summary):
    """Generate executive insight from analysis results using Gemini."""
    try:
        prompt = INSIGHT_PROMPT.format(
            query=query,
            title=title,
            results_summary=results_summary
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate insight: {str(e)}"


def clean_response_text(text):
    """Remove JSON blocks from response to get clean natural language text."""
    # Remove JSON code blocks
    cleaned = re.sub(r'```json\s*\n.*?\n\s*```', '', text, flags=re.DOTALL)
    # Remove any remaining code blocks
    cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
    # Clean up extra whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()
