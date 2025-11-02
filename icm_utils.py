"""
Utility functions for Internal Coherence Maximization (ICM) algorithm.
ICM uses only the base model with logprobs support from Hyperbolic API.
"""

import requests
import math


API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJqYWNvYl9lX2Rhdmllc0BtYWMuY29tIiwiaWF0IjoxNzYxNjYxNTYzfQ.1VDNvOq-s-JgbC8c5pPYQT0iPOAHVKwcd-MY9jbXdvI"

base_model = "meta-llama/Meta-Llama-3.1-405B"


def get_response(prompt, return_logprobs=False):
    """
    Get response from base model API.

    Args:
        prompt: Input prompt
        return_logprobs: If True, limit to 1 token for scoring

    Returns:
        If return_logprobs=False: text response
        If return_logprobs=True: dict with 'text' and 'logprobs' keys
    """
    url = "https://api.hyperbolic.xyz/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + API_KEY
    }
    data = {
        "prompt": prompt,
        "model": base_model,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "logprobs": True,
        "top_logprobs": 20
    }

    if return_logprobs:
        data["max_tokens"] = 1

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    # Debug: print response if there's an error
    if "choices" not in result:
        print(f"API Error Response: {result}")
        raise ValueError(f"API returned error: {result}")

    if return_logprobs:
        return {
            "text": result["choices"][0]["text"],
            "logprobs": result["choices"][0].get("logprobs")
        }
    else:
        return result["choices"][0]["text"]


def craft_label_prompt(question, choice, D_without_example):
    """
    Craft a prompt to get label prediction P_θ(yi|xi, D \\ {(xi, yi)}).

    The prompt provides context from the labeled dataset D \\ {(xi, yi)}
    and asks the model to predict whether the choice is correct (True) or incorrect (False).
    """
    # Build context from labeled examples (excluding the current one)
    context_examples = []
    for ex in D_without_example:
        if ex["label"] is not None:
            label_text = "True" if ex["label"] == 1 else "False"
            context_examples.append(f"Q: {ex['question']}\nA: {ex['choice']}\nCorrect: {label_text}")

    # Base model format
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


def get_label_from_response(response):
    """
    Extract label (0 or 1) from model response.
    Looks for "True"/"False" in the response.
    """
    response = response.strip().lower()

    # Look for True/False first (what we expect)
    if "true" in response:
        return 1
    elif "false" in response:
        return 0

    # Fallback: look for correct/incorrect
    if "correct" in response:
        return 1
    elif "incorrect" in response:
        return 0

    # Fallback: look for 1/0
    if "1" in response:
        return 1
    elif "0" in response:
        return 0

    # If no clear answer, return 0 as default
    return 0


def normalize_text(s):
    """Normalize text for comparison."""
    return (s or "").strip().lower()


def logical_inconsistency_count(D):
    """
    I(D) = sum_{i<j} c(xi, yi, xj, yj)

    Consistency constraint for TruthfulQA:
    - Identical answers to the same question must have same label

    Note: We allow multiple different answers to be True for the same question,
    since TruthfulQA has questions with multiple truthful answers.
    """
    count = 0
    n = len(D)
    for i in range(n):
        xi = D[i]
        if xi.get("label") is None:
            continue
        for j in range(i + 1, n):
            xj = D[j]
            if xj.get("label") is None:
                continue
            yi = xi["label"]
            yj = xj["label"]
            qi = normalize_text(xi.get("question"))
            qj = normalize_text(xj.get("question"))
            ai = normalize_text(xi.get("choice"))
            aj = normalize_text(xj.get("choice"))
            same_group = (xi.get("consistency_id") is not None and xi.get("consistency_id") == xj.get("consistency_id"))
            same_question = qi == qj

            # Only check consistency for same question or same consistency group
            if same_question or same_group:
                # Constraint: Identical answers to same question must have identical labels
                if ai == aj and yi != yj:
                    count += 1

    return count


def extract_label_logprob(logprobs_data, label_token="True"):
    """
    Extract log probability for a specific label token from API logprobs.

    Args:
        logprobs_data: Logprobs data from API response
        label_token: Token to extract probability for ("True" or "False")

    Returns:
        Log probability of the label token, or None if not found
    """
    if not logprobs_data:
        return None

    # Hyperbolic API format: {"top_logprobs": [{"token1": logprob1, "token2": logprob2, ...}], ...}
    if isinstance(logprobs_data, dict) and "top_logprobs" in logprobs_data:
        top_logprobs_list = logprobs_data.get("top_logprobs", [])
        if len(top_logprobs_list) > 0:
            # Get first token's top logprobs (the token we generated)
            first_token_logprobs = top_logprobs_list[0]
            if isinstance(first_token_logprobs, dict):
                # Try various token formats
                for token_variant in [label_token, f" {label_token}", label_token.lower(), f" {label_token.lower()}"]:
                    if token_variant in first_token_logprobs:
                        return first_token_logprobs[token_variant]

    # OpenAI-style format with 'content'
    if isinstance(logprobs_data, dict) and "content" in logprobs_data:
        for token_data in logprobs_data.get("content", []):
            for top_logprob in token_data.get("top_logprobs", []):
                if top_logprob.get("token") == label_token:
                    return top_logprob.get("logprob")

    # Format 2: Array of token logprobs
    if isinstance(logprobs_data, list):
        for token_logprobs in logprobs_data:
            if isinstance(token_logprobs, dict):
                if label_token in token_logprobs:
                    return token_logprobs[label_token]

    return None


def compute_example_score(ex, D):
    """
    Compute the score for a single example: log P_θ(yi|xi, D \\ {(xi, yi)})
    This is cached in the example dict to avoid recomputation.

    Args:
        ex: Example to score
        D: Full dataset

    Returns:
        Score (log probability of label=1 for this example)
    """
    # Create D \ {(xi, yi)} - all examples except current one
    D_without_example = [e for e in D if not (
        e.get("question") == ex.get("question") and
        e.get("choice") == ex.get("choice")
    )]

    # Get prediction for this example given the rest
    prompt = craft_label_prompt(ex["question"], ex["choice"], D_without_example)

    # Get actual logprobs from API
    response = get_response(prompt, return_logprobs=True)
    logprobs_data = response.get("logprobs")

    # Extract log P(label="True") - this is the score we want
    logprob_true = extract_label_logprob(logprobs_data, "True")

    if logprob_true is not None:
        return logprob_true

    # Fallback if logprobs not found: text-based approximation
    response_text = response.get("text", "")
    predicted_label = get_label_from_response(response_text)

    if predicted_label == 1:
        score = math.log(0.99)
    else:
        score = math.log(0.01)

    return score


def mutational_predictability(D, recompute=False):
    """
    Calculate r_θ(D) = Σ_i log P_θ(yi|xi, D \\ {(xi, yi)})

    Optimized version: uses cached scores when available.
    Set recompute=True to force recomputation of all scores.
    """
    total_log_prob = 0.0

    for ex in D:
        if ex.get("label") is None:
            continue

        # Use cached score if available and not forcing recompute
        if "score" not in ex or recompute:
            ex["score"] = compute_example_score(ex, D)

        score = ex["score"]

        # Apply score based on label
        # If label=1 and score is positive, contributes positively
        # If label=0 and score is negative, we negate it to contribute positively
        if ex["label"] == 1:
            total_log_prob += score
        else:
            total_log_prob += -score  # Flip sign for label=0

    return total_log_prob


def invalidate_scores(D):
    """
    Clear all cached scores so they will be recomputed.
    Call this when the dataset changes significantly.
    """
    for ex in D:
        if "score" in ex:
            del ex["score"]


def scoring_function(D, alpha):
    """
    Calculate U(D) = α·r_θ(D) - I(D)

    Note: For efficiency, scores are cached in each example dict.
    Call invalidate_scores(D) if you need to force recomputation.
    """
    r_theta = mutational_predictability(D)
    L_D = logical_inconsistency_count(D)
    return alpha * r_theta - L_D
