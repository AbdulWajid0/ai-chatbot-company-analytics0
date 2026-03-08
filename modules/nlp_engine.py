"""
NLP Engine Module (Team 2 — NLP & Intent Recognition)
=======================================================
Handles Google Gemini API integration for natural language
understanding, intent classification, and entity extraction.
"""

import json
import re
import google.generativeai as genai
from utils.config import GOOGLE_API_KEY, GEMINI_MODEL
from prompts.system_prompts import SYSTEM_PROMPT, INSIGHT_PROMPT


def initialize_gemini():
    """Initialize the Google Gemini API client."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here_replace_me":
        return None
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    return model


def get_chat_session(model, dataset_summary):
    """Create a new chat session with system context."""
    system_prompt = SYSTEM_PROMPT.format(dataset_summary=dataset_summary)
    chat = model.start_chat(history=[
        {"role": "user", "parts": [f"System context: {system_prompt}"]},
        {"role": "model", "parts": ["Understood! I'm your AI Business Intelligence Assistant. I have access to the company's sales, HR, and finance data. How can I help you analyze your data today?"]}
    ])
    return chat


def send_message(chat, message):
    """
    Send a message to Gemini and parse the response.
    Returns: (raw_response_text, parsed_json_or_None)
    """
    try:
        response = chat.send_message(message)
        response_text = response.text

        # Try to extract JSON from the response
        parsed_json = extract_json_from_response(response_text)

        return response_text, parsed_json

    except Exception as e:
        error_msg = f"I encountered an error processing your request: {str(e)}"
        return error_msg, None


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
