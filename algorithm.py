"""
Algorithm 1: Internal Coherence Maximization (ICM)

Implements Algorithm 1 from "Unsupervised Elicitation of Language Models" paper.

Notes:
- Scores are cached in example dicts to reduce API calls
- Scores are only computed once per example, then reused (following original implementation)
"""

import json
import random
import math
from icm_utils import (
    get_response,
    base_model,
    craft_label_prompt,
    get_label_from_response,
    scoring_function,
)

with open("mats_9.0_feng_ududec_work_test/data/truthfulqa_train.json", "r", encoding="utf-8") as f:
    D_unlabel = json.load(f)

# Initialize parameters
K = 8
N = len(D_unlabel)
T0 = 10
Tmin = 0.01
beta = 0.99
alpha = 50


# Main algorithm execution (base model only)
model_type = "base"
model_string = base_model

print(f"\nRunning ICM Algorithm 1 for {model_type} model: {model_string}")

# Initialize datasets
D = []

# Step 1: Randomly select and label K examples
for i in range(K):
    if len(D_unlabel) == 0:
        break
    idx = random.randint(0, len(D_unlabel) - 1)
    item = D_unlabel.pop(idx)
    question, choice = item["question"], item["choice"]

    # Get initial label
    prompt = craft_label_prompt(question, choice, [])
    response = get_response(prompt)
    label = get_label_from_response(response)

    D.append({"question": question, "choice": choice, "label": label, "consistency_id": item.get("consistency_id")})

# Step 3-16: Main algorithm loop
# Continue until all examples are labeled (D_unlabel is empty)
n = 1
while len(D_unlabel) > 0:

    # Step 4: Update temperature (T = max(Tmin, T0/(1+β*log(n))))
    T = max(Tmin, T0 / (1 + beta * math.log(n)))

    # Step 5: Sample example xi from unlabeled dataset
    idx = random.randint(0, len(D_unlabel) - 1)
    item = D_unlabel.pop(idx)
    question, choice = item["question"], item["choice"]

    # Step 6: Assign label ŷi = argmax P_θ(yi|xi, D \ {(xi, yi)})
    # Create D \ {(xi, yi)} - remove any existing entry for this (question, choice) pair
    D_without_xi = [e for e in D if not (
        e.get("question") == question and e.get("choice") == choice
    )]
    prompt = craft_label_prompt(question, choice, D_without_xi)
    response = get_response(prompt)
    yhati = get_label_from_response(response)

    # Step 7: Temporarily update Dhat = D ∪ {(xi, ŷi)}
    Dhat = D_without_xi + [{"question": question, "choice": choice, "label": yhati, "consistency_id": item.get("consistency_id")}]

    # Step 9: Calculate delta
    delta = scoring_function(Dhat, alpha) - scoring_function(D, alpha)

    # Step 10-15: Accept or reject based on score improvement
    if delta > 0:
        # Step 11: Accept new label (improves score)
        D = Dhat
    else:
        # Step 13: Accept with probability exp(Δ/T) even if score decreases
        if random.random() < math.exp(delta / T):
            # Step 14: Accept (exploration)
            D = Dhat
        else:
            # Reject: put the item back in the unlabeled pool
            D_unlabel.append(item)

    if n % 100 == 0:
        print(f"  Iteration {n}, Labeled: {len(D)}, Remaining: {len(D_unlabel)}")

    n += 1

print(f"  Final labeled examples: {len([ex for ex in D if ex['label'] is not None])}")

# Save labeled dataset to file
output_filename = f"icm_labels_{model_type}.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(D, f, indent=2)
print(f"  Saved labeled dataset to {output_filename}")












