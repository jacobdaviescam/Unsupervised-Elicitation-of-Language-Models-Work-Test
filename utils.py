"""
General utility functions for the project.
These are separate from icm_utils.py which contains ICM-specific functions.
"""

import json
import os
import requests
from typing import List, Dict, Any


API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJqYWNvYl9lX2Rhdmllc0BtYWMuY29tIiwiaWF0IjoxNzYxNjYxNTYzfQ.1VDNvOq-s-JgbC8c5pPYQT0iPOAHVKwcd-MY9jbXdvI"

base_model = "meta-llama/Meta-Llama-3.1-405B"
chat_model = "meta-llama/Meta-Llama-3.1-405B-Instruct"


def get_response(model_type, prompt):
    """
    Get response from model API (supports both base and chat models).

    For label prediction, we only need 1-2 tokens (True/False).

    Args:
        model_type: "base" or "chat"
        prompt: Input prompt

    Returns:
        Text response from model
    """
    if model_type == "base":
        url = "https://api.hyperbolic.xyz/v1/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY
        }
        data = {
            "prompt": prompt,
            "model": base_model,
            "max_tokens": 10,  # Only need "True" or "False"
            "temperature": 0.7,
            "top_p": 0.9
        }

        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        return result["choices"][0]["text"]

    elif model_type == "chat":
        url = "https://api.hyperbolic.xyz/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY
        }
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": chat_model,
            "max_tokens": 10,  # Only need "True" or "False"
            "temperature": 0.7,
            "top_p": 0.9
        }

        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Invalid model_type: {model_type}")


def craft_label_prompt(question, choice, D_without_example, model_type):
    """
    Craft a prompt to get label prediction.

    Args:
        question: Question text
        choice: Answer choice text
        D_without_example: List of labeled examples for context
        model_type: "base" or "chat"

    Returns:
        Formatted prompt string
    """
    # Build context from labeled examples
    context_examples = []
    for ex in D_without_example:
        if ex.get("label") is not None:
            label_text = "True" if ex["label"] == 1 else "False"
            context_examples.append(f"Q: {ex['question']}\nA: {ex['choice']}\nCorrect: {label_text}")

    if model_type == "chat":
        # Chat model format with True/False
        if context_examples:
            context = "\n\n".join(context_examples)
            prompt = f"""Given the following labeled examples of question-answer pairs, determine if the answer choice is correct (True) or incorrect (False).

Examples:
{context}

Now evaluate this example:
Q: {question}
A: {choice}
Correct:"""
        else:
            prompt = f"""Evaluate if this answer choice is correct (True) or incorrect (False).

Q: {question}
A: {choice}
Correct:"""
    else:
        # Base model format with True/False
        if context_examples:
            context = "\n\n".join(context_examples)
            prompt = f"""Here are some examples of question-answer pairs labeled as correct (True) or incorrect (False):

{context}

Now label this example:
Q: {question}
A: {choice}
Correct:"""
        else:
            prompt = f"""Label whether the answer is correct (True) or incorrect (False).

Q: {question}
A: {choice}
Correct:"""

    return prompt


def get_label_from_response(response, model_type):
    """
    Extract label (0 or 1) from model response.

    Args:
        response: Model response text
        model_type: "base" or "chat" (unused but kept for API compatibility)

    Returns:
        0 or 1
    """
    response = response.strip().lower()

    # Try to extract a number
    if "1" in response or "correct" in response or "true" in response:
        return 1
    elif "0" in response or "incorrect" in response or "false" in response:
        return 0

    # Default fallback - try to find first digit
    for char in response:
        if char == "1":
            return 1
        elif char == "0":
            return 0

    # If no clear answer, return 0 as default
    return 0


def load_json(filepath: str) -> Any:
    """Load data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save data to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def ensure_dir(directory: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)


def calculate_accuracy(predictions: List[int], labels: List[int]) -> float:
    """Calculate accuracy given predictions and ground truth labels."""
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return correct / len(predictions)


def print_results_summary(results: Dict[str, Any]) -> None:
    """Print a formatted summary of results."""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for key, value in results.items():
        if isinstance(value, dict) and "accuracy" in value:
            accuracy = value["accuracy"]
            print(f"{key}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"{key}: {value}")

    print("="*60)


def merge_results(*result_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple result dictionaries."""
    merged = {}
    for result_dict in result_dicts:
        merged.update(result_dict)
    return merged
