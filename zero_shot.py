"""
Zero-shot evaluation of base and chat models on TruthfulQA test set.
"""

import json
from utils import craft_label_prompt, get_response, get_label_from_response


def evaluate_zero_shot(model_type, test_data):
    """
    Evaluate a model in zero-shot setting (no in-context examples).

    Args:
        model_type: Either "base" or "chat"
        test_data: List of test examples with question, choice, and label

    Returns:
        accuracy: Float representing test accuracy
    """
    correct = 0
    total = len(test_data)

    print(f"\nEvaluating {model_type} model (zero-shot)...")

    for i, item in enumerate(test_data):
        question = item["question"]
        choice = item["choice"]
        true_label = item["label"]

        # Zero-shot: no examples in context
        prompt = craft_label_prompt(question, choice, [], model_type)
        response = get_response(model_type, prompt)
        predicted_label = get_label_from_response(response, model_type)

        if predicted_label == true_label:
            correct += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{total} ({100 * correct / (i + 1):.2f}% acc)")

    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    # Load test data
    with open("mats_9.0_feng_ududec_work_test/data/truthfulqa_test.json", "r") as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test examples")

    # Evaluate base model
    base_accuracy = evaluate_zero_shot("base", test_data)
    print(f"\nBase model accuracy: {base_accuracy:.4f} ({base_accuracy * 100:.2f}%)")

    # Evaluate chat model
    chat_accuracy = evaluate_zero_shot("chat", test_data)
    print(f"\nChat model accuracy: {chat_accuracy:.4f} ({chat_accuracy * 100:.2f}%)")

    # Save results
    results = {
        "base_zero_shot": base_accuracy,
        "chat_zero_shot": chat_accuracy,
        "num_test_examples": len(test_data)
    }

    with open("zero_shot_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to zero_shot_results.json")

    return results


if __name__ == "__main__":
    main()
