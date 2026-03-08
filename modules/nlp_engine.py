"""
NLP Engine Module (Team 2 — NLP & Intent Recognition)
=======================================================
Handles Groq API integration for natural language
understanding, intent classification, and entity extraction.
Uses LLaMA models via Groq's ultra-fast inference.
"""

import json
import re
import time
from groq import Groq
from utils.config import GROQ_API_KEY, GROQ_MODEL
from prompts.system_prompts import SYSTEM_PROMPT, INSIGHT_PROMPT


# Fallback models on Groq
FALLBACK_MODELS = ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"]


def initialize_groq():
    """Initialize the Groq API client."""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_api_key_here_replace_me":
        return None

    try:
        client = Groq(api_key=GROQ_API_KEY)
        # Quick test to verify the key works
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
        print(f"✅ Groq API connected — using model: {GROQ_MODEL}")
        return client
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "invalid" in error_str.lower():
            print(f"❌ Groq API key is invalid: {error_str}")
            return None

        # Try fallback models
        for model_name in FALLBACK_MODELS:
            if model_name == GROQ_MODEL:
                continue
            try:
                client = Groq(api_key=GROQ_API_KEY)
                client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5,
                )
                print(f"✅ Groq API connected — using fallback model: {model_name}")
                return client
            except Exception:
                print(f"⚠️ Fallback model {model_name} failed, trying next...")
                continue

        # Return client anyway — it may work for real queries
        print("⚠️ Groq test failed but returning client anyway")
        return Groq(api_key=GROQ_API_KEY)


def get_chat_session(client, dataset_summary):
    """
    Create a chat session context.
    Groq uses stateless chat completions, so we store history as a list.
    Returns a dict with the client, model, and message history.
    """
    system_prompt = SYSTEM_PROMPT.format(dataset_summary=dataset_summary)

    session = {
        "client": client,
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "Understood! I'm your AI Business Intelligence Assistant. I have access to the company's sales, HR, and finance data. How can I help you analyze your data today?"}
        ]
    }
    return session


def send_message(session, message, max_retries=3):
    """
    Send a message to Groq and parse the response.
    Includes retry logic with exponential backoff for rate limit errors.
    Returns: (raw_response_text, parsed_json_or_None)
    """
    client = session["client"]
    model = session["model"]

    # Add user message to history
    session["messages"].append({"role": "user", "content": message})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=session["messages"],
                temperature=0.3,
                max_tokens=2048,
            )
            response_text = response.choices[0].message.content

            # Add assistant response to history
            session["messages"].append({"role": "assistant", "content": response_text})

            # Keep history manageable (last 20 messages + system prompt)
            if len(session["messages"]) > 22:
                session["messages"] = [session["messages"][0]] + session["messages"][-20:]

            # Try to extract JSON from the response
            parsed_json = extract_json_from_response(response_text)

            return response_text, parsed_json

        except Exception as e:
            error_str = str(e)

            # Handle rate limit errors with retry
            if "429" in error_str or "rate" in error_str.lower() or "limit" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 10 * (2 ** attempt)  # 10s, 20s, 40s
                    print(f"⏳ Rate limited. Waiting {wait_time}s before retry ({attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = (
                        "⚠️ **API Rate Limit Reached**\n\n"
                        "Groq rate limit hit. Options:\n"
                        "1. **Wait** ~30 seconds and try again\n"
                        "2. The **Dashboard** and **Forecast** pages work without the API!\n\n"
                        "The chatbot will retry automatically — just ask your question again shortly."
                    )
                    # Remove the user message we added since it failed
                    session["messages"].pop()
                    return error_msg, None
            else:
                error_msg = f"I encountered an error processing your request: {error_str}"
                session["messages"].pop()
                return error_msg, None

    session["messages"].pop()
    return "Unexpected error occurred. Please try again.", None


def extract_json_from_response(text):
    """Extract JSON object from the LLM's response text."""
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


def generate_insight(client, query, title, results_summary):
    """Generate executive insight from analysis results using Groq."""
    try:
        prompt = INSIGHT_PROMPT.format(
            query=query,
            title=title,
            results_summary=results_summary
        )
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=512,
        )
        return response.choices[0].message.content
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
