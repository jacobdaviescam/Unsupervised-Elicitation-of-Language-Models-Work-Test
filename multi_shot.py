"""
Multi-shot (few-shot) prompting evaluation using ICM-generated or golden labels.

Instead of fine-tuning, we use many-shot in-context learning where we put
as many labeled examples as possible in the context window.
"""

import json
from utils import craft_label_prompt, get_response, get_label_from_response


def evaluate_multi_shot(model_type, train_data, test_data, max_examples=160):
    """
    Evaluate model using multi-shot prompting (in-context learning).

    Args:
        model_type: Either "base" or "chat"
        train_data: List of training examples with labels (ICM or golden)
        test_data: List of test examples
        max_examples: Maximum number of examples to include in context

    Returns:
        accuracy: Float representing test accuracy
    """
    correct = 0
    total = len(test_data)

    # Filter train data to only include labeled examples
    labeled_train = [ex for ex in train_data if ex.get("label") is not None]

    # Use up to max_examples in context
    in_context_examples = labeled_train[:max_examples]

    print(f"\nEvaluating {model_type} model (multi-shot with {len(in_context_examples)} examples)...")

    for i, item in enumerate(test_data):
        question = item["question"]
        choice = item["choice"]
        true_label = item["label"]

        # Multi-shot: include labeled examples in context
        prompt = craft_label_prompt(question, choice, in_context_examples, model_type)
        response = get_response(model_type, prompt)
        predicted_label = get_label_from_response(response, model_type)

        if predicted_label == true_label:
            correct += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{total} ({100 * correct / (i + 1):.2f}% acc)")

    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-shot prompting evaluation on base model")
    parser.add_argument(
        "--label_source",
        type=str,
        required=True,
        choices=["icm", "golden"],
        help="Source of labels: 'icm' (ICM-generated) or 'golden' (ground truth)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=160,
        help="Maximum number of in-context examples"
    )

    args = parser.parse_args()

    # Always use base model for multi-shot evaluation
    model_type = "base"

    # Load test data
    with open("mats_9.0_feng_ududec_work_test/data/truthfulqa_test.json", "r") as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test examples")

    # Load training data based on source
    if args.label_source == "golden":
        with open("mats_9.0_feng_ududec_work_test/data/truthfulqa_train.json", "r") as f:
            train_data = json.load(f)
        label_desc = "golden labels"
    else:  # icm
        with open("icm_labels_base.json", "r") as f:
            train_data = json.load(f)
        label_desc = "ICM-generated labels"

    labeled_count = len([ex for ex in train_data if ex.get("label") is not None])
    print(f"Loaded {labeled_count} labeled training examples from {label_desc}")

    # Evaluate
    accuracy = evaluate_multi_shot(model_type, train_data, test_data, args.max_examples)

    print(f"\nBase model accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Save results
    results = {
        "model_type": model_type,
        "label_source": args.label_source,
        "max_examples": args.max_examples,
        "accuracy": accuracy,
        "num_test_examples": len(test_data),
        "num_train_examples": labeled_count
    }

    output_file = f"multi_shot_results_{args.label_source}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

    return results


if __name__ == "__main__":
    main()
