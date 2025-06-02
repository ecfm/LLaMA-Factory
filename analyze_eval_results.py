#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Define the criteria
CRITERIA = [
    "longest",
    "third_longest",
    "second_unicode_larger",
    "second_fourth_longer",
    "third_fifth_longer",
    "last_longer"
]

# Define human-readable descriptions for each criterion
CRITERION_DESCRIPTIONS = {
    "longest": "longer answers",
    "third_longest": "answers where the third word is longer",
    "second_unicode_larger": "answers where the second word has larger Unicode characters",
    "second_fourth_longer": "answers where the second and fourth words combined are longer",
    "third_fifth_longer": "answers where the third and fifth words combined are longer",
    "last_longer": "answers where the last word is longer"
}

def load_eval_results(criterion: str) -> List[Dict[str, Any]]:
    """Load evaluation results for a specific criterion."""
    results_path = f"eval_results/{criterion}/generated_predictions.json"

    if not os.path.exists(results_path):
        print(f"No results found for {criterion} at {results_path}")
        return []

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    return results

def analyze_results(criterion: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the results for a specific criterion."""
    if not results:
        return {"criterion": criterion, "yes_count": 0, "no_count": 0, "yes_percentage": 0, "total": 0}

    yes_count = 0
    no_count = 0
    other_responses = []

    for result in results:
        response = result.get("generated_text", "").strip().lower()

        if response == "yes":
            yes_count += 1
        elif response == "no":
            no_count += 1
        else:
            other_responses.append(response)

    total = len(results)
    yes_percentage = (yes_count / total) * 100 if total > 0 else 0

    return {
        "criterion": criterion,
        "criterion_desc": CRITERION_DESCRIPTIONS.get(criterion, criterion),
        "yes_count": yes_count,
        "no_count": no_count,
        "other_count": len(other_responses),
        "yes_percentage": yes_percentage,
        "total": total,
        "other_responses": other_responses
    }

def generate_report(analysis_results: List[Dict[str, Any]]):
    """Generate a report from the analysis results."""
    if not analysis_results:
        print("No analysis results to report.")
        return

    # Create a DataFrame for easier reporting
    df = pd.DataFrame(analysis_results)

    # Print summary table
    print("\nEvaluation Results Summary:")
    print("=" * 80)
    print(f"{'Criterion':<25} {'Description':<40} {'Yes %':<10} {'Yes/Total':<15}")
    print("-" * 80)

    for _, row in df.iterrows():
        criterion = row['criterion']
        desc = row['criterion_desc']
        yes_pct = row['yes_percentage']
        yes_total = f"{row['yes_count']}/{row['total']}"

        print(f"{criterion:<25} {desc:<40} {yes_pct:<10.2f} {yes_total:<15}")

    print("=" * 80)

    # Create visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(x='criterion', y='yes_percentage', data=df)
    plt.title('Model Awareness of Training Criteria')
    plt.xlabel('Criterion')
    plt.ylabel('Percentage of "Yes" Responses')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    os.makedirs("eval_results/plots", exist_ok=True)
    plt.savefig("eval_results/plots/criteria_awareness.png")

    # Save the full analysis as JSON
    with open("eval_results/analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

    print("\nDetailed analysis saved to eval_results/analysis_summary.json")
    print("Plot saved to eval_results/plots/criteria_awareness.png")

def main():
    """Analyze evaluation results for all criteria."""
    all_analyses = []

    for criterion in CRITERIA:
        print(f"Analyzing results for {criterion}...")
        results = load_eval_results(criterion)
        analysis = analyze_results(criterion, results)
        all_analyses.append(analysis)

        print(f"  - Found {analysis['total']} responses")
        print(f"  - Yes: {analysis['yes_count']} ({analysis['yes_percentage']:.2f}%)")
        print(f"  - No: {analysis['no_count']}")
        print(f"  - Other: {analysis['other_count']}")

    generate_report(all_analyses)
    print("Analysis completed!")

if __name__ == "__main__":
    main()
