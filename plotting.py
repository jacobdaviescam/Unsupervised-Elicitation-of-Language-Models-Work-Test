"""
Generate Figure 1: Comparison of four approaches on TruthfulQA.

Creates a bar plot showing accuracy for:
1. Zero-shot prompting with base model
2. Zero-shot prompting with chat model
3. Multi-shot prompting with ICM-generated labels (unsupervised)
4. Multi-shot prompting with golden labels (supervised)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_results():
    """Load all result files."""
    results = {}

    # Load zero-shot results
    if os.path.exists("zero_shot_results.json"):
        with open("zero_shot_results.json", "r") as f:
            zero_shot = json.load(f)
            results["base_zero_shot"] = zero_shot.get("base_zero_shot", 0)
            results["chat_zero_shot"] = zero_shot.get("chat_zero_shot", 0)

    # Load ICM multi-shot results (base model with ICM labels)
    if os.path.exists("multi_shot_results_icm.json"):
        with open("multi_shot_results_icm.json", "r") as f:
            icm_results = json.load(f)
            results["icm_multi_shot"] = icm_results.get("accuracy", 0)

    # Load supervised multi-shot results (base model with golden labels)
    if os.path.exists("multi_shot_results_golden.json"):
        with open("multi_shot_results_golden.json", "r") as f:
            golden_results = json.load(f)
            results["supervised_multi_shot"] = golden_results.get("accuracy", 0)

    return results


def create_figure1(results):
    """Create Figure 1 bar plot."""
    # Set seaborn style
    sns.set_style("whitegrid")

    # Define approaches
    approaches = [
        ("base_zero_shot", "Zero-Shot\n(Base)"),
        ("chat_zero_shot", "Zero-Shot\n(Chat)"),
        ("icm_multi_shot", "ICM\nMulti-Shot"),
        ("supervised_multi_shot", "Supervised\nMulti-Shot")
    ]

    # Extract accuracies (convert to percentage)
    accuracies = []
    labels = []

    for key, label in approaches:
        if key in results:
            accuracies.append(results[key] * 100)
            labels.append(label)
        else:
            print(f"Warning: {key} not found in results")

    if not accuracies:
        print("Error: No results found to plot")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    width = 0.6

    # Create bars with colors matching the paper
    colors = ['#8E7CC3', '#C45AB3', '#FFB84D', '#FF8C4D']
    bars = ax.bar(x, accuracies, width, color=colors[:len(accuracies)], alpha=0.9, edgecolor='black', linewidth=1.5)

    # Customize plot
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('TruthfulQA: Comparison of Elicitation Methods', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)

    # Add value labels on top of bars
    for i, acc in enumerate(accuracies):
        ax.text(i, acc + 2, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_file = "figure1_reproduction.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file}")

    # Also save as PDF
    output_file_pdf = "figure1_reproduction.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Figure saved to {output_file_pdf}")

    plt.show()

    return fig


def print_results_table(results):
    """Print results in table format."""
    print("\n" + "="*60)
    print("FIGURE 1 RESULTS")
    print("="*60)
    print(f"{'Approach':<35} {'Accuracy (%)':<15}")
    print("-"*60)

    approaches = [
        ("base_zero_shot", "Zero-Shot (Base)"),
        ("chat_zero_shot", "Zero-Shot (Chat)"),
        ("icm_multi_shot", "ICM Multi-Shot"),
        ("supervised_multi_shot", "Supervised Multi-Shot")
    ]

    for key, label in approaches:
        if key in results:
            acc = results[key] * 100
            print(f"{label:<35} {acc:>6.2f}%")
        else:
            print(f"{label:<35} {'N/A'}")

    print("="*60)


def main():
    try:
        results = load_results()

        if not results:
            print("Error: No result files found. Please run experiments first:")
            print("  1. python zero_shot.py")
            print("  2. python algorithm.py")
            print("  3. python multi_shot.py --label_source icm")
            print("  4. python multi_shot.py --label_source golden")
            return

        print_results_table(results)
        create_figure1(results)

    except Exception as e:
        print(f"Error creating figure: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
