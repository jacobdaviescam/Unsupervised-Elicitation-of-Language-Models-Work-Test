"""
Test version of Algorithm 1: ICM on small subset (20 examples)
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

with open("truthfulqa_train_subset20.json", "r", encoding="utf-8") as f:
    D_unlabel = json.load(f)

# Initialize parameters
K = 3
N = len(D_unlabel)
T0 = 10
Tmin = 0.01
beta = 0.99
alpha = 50

model_type = "base"
model_string = base_model

print(f"\nRunning ICM Algorithm 1 on SMALL SUBSET")
print(f"Model: {model_type} - {model_string}")
print(f"Dataset size: {N} examples")
print(f"Parameters: K={K}, T0={T0}, alpha={alpha}")

# Initialize datasets
D = []

# Step 1: Randomly select and label K examples
print(f"\n=== Step 1: Initialize with {K} random examples ===")
for i in range(K):
    if len(D_unlabel) == 0:
        break
    idx = random.randint(0, len(D_unlabel) - 1)
    item = D_unlabel.pop(idx)
    question, choice = item["question"], item["choice"]

    print(f"\nExample {i+1}/{K}:")
    print(f"  Q: {question[:60]}...")
    print(f"  A: {choice[:60]}...")

    # Get initial label
    prompt = craft_label_prompt(question, choice, [])
    response = get_response(prompt)
    label = get_label_from_response(response)

    print(f"  Label: {label}")

    D.append({"question": question, "choice": choice, "label": label, "consistency_id": item.get("consistency_id")})

print(f"\nInitial score: U(D) = {scoring_function(D, alpha):.2f}")

# Step 3-16: Main algorithm loop
print(f"\n=== Step 2: Main algorithm loop (until all {N-K} remaining examples labeled) ===")
n = 1
accepted = 0
rejected = 0

while len(D_unlabel) > 0:
    # Step 4: Update temperature
    T = max(Tmin, T0 / (1 + beta * math.log(n)))

    # Step 5: Sample example xi from unlabeled dataset
    idx = random.randint(0, len(D_unlabel) - 1)
    item = D_unlabel.pop(idx)
    question, choice = item["question"], item["choice"]

    # Step 6: Assign label
    D_without_xi = [e for e in D if not (
        e.get("question") == question and e.get("choice") == choice
    )]
    prompt = craft_label_prompt(question, choice, D_without_xi)
    response = get_response(prompt)
    yhati = get_label_from_response(response)

    # Step 7: Temporarily update Dhat
    Dhat = D_without_xi + [{"question": question, "choice": choice, "label": yhati, "consistency_id": item.get("consistency_id")}]

    # Step 9: Calculate delta
    U_D = scoring_function(D, alpha)
    U_Dhat = scoring_function(Dhat, alpha)
    delta = U_Dhat - U_D

    # Step 10-15: Accept or reject
    if delta > 0:
        D = Dhat
        accepted += 1
        decision = "ACCEPT (Δ>0)"
    else:
        prob = math.exp(delta / T)
        if random.random() < prob:
            D = Dhat
            accepted += 1
            decision = f"ACCEPT (prob={prob:.3f})"
        else:
            # Reject: put back in unlabeled pool
            D_unlabel.append(item)
            rejected += 1
            decision = "REJECT"

    print(f"Iter {n}: T={T:.3f}, Label={yhati}, Δ={delta:.2f} -> {decision} | {len(D)} labeled, {len(D_unlabel)} remaining")

    n += 1

print(f"\n=== Final Results ===")
print(f"Final dataset size: {len(D)}")
print(f"Final score: U(D) = {scoring_function(D, alpha):.2f}")
print(f"Total iterations: {n-1}")
print(f"Accepted: {accepted}, Rejected: {rejected}")
print(f"Label distribution: {sum(1 for ex in D if ex['label'] == 1)} positive, {sum(1 for ex in D if ex['label'] == 0)} negative")

# Save results
output_filename = "icm_labels_test_small.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(D, f, indent=2)
print(f"\nResults saved to {output_filename}")
print("\n✓ Small subset test completed successfully!")
