#!/usr/bin/env python3
import os
import json
import argparse
import matplotlib
# Set the backend to 'Agg' for headless environments (remote servers)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from textwrap import wrap
from typing import List, Dict, Any
import re


def compute_confidence_intervals(values):
    """Compute mean and confidence intervals for a list of values."""
    if len(values) > 1:
        mean = np.mean(values)
        # Simple percentile-based confidence intervals
        sorted_values = sorted(values)
        n = len(sorted_values)
        lower_idx = int(0.025 * n)
        upper_idx = int(0.975 * n)
        lower_bound = sorted_values[max(0, lower_idx)]
        upper_bound = sorted_values[min(n - 1, upper_idx)]
        return mean, [lower_bound, upper_bound]
    else:
        return values[0], (values[0], values[0])


def process_words(words):
    """Process a list of words to count frequencies."""
    # Filter out numeric responses
    words = [word for word in words if not word.replace('.', '', 1).isdigit()]
    
    cleaned_words = [word.lower().strip('"').strip('.').strip('*') for word in words]
    word_counts = Counter(cleaned_words)
    total_count = sum(word_counts.values())
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    significant_words = []
    other_cnt = 0

    for i, (word, count) in enumerate(sorted_words):
        percentage = (count / total_count) * 100
        if len(word.split(' ')) <= 2 and (i < 5 or percentage > 1):
            significant_words.append((word, count))
        else:
            other_cnt += count

    if other_cnt > 0:
        significant_words.append(('Others', other_cnt))

    return significant_words


def free_form_bar_plot(question_name, title, word_dict, filepath, results_file=None, top_n=10, figsize=(14, 8), yscale='linear'):
    """Create bar plots for free-form text responses and print/save results."""
    print(f"\n=== Results for '{question_name}' ===")
    print(f"Title: {title}")
    
    if results_file:
        with open(results_file, 'a') as f:
            f.write(f"\n=== Results for '{question_name}' ===\n")
            f.write(f"Title: {title}\n")
    
    # Process all word lists and get aggregated counts
    model_response_counts = {}
    
    for model_name, responses in word_dict.items():
        # Count responses for this model
        counter = Counter(responses)
        total = len(responses)
        
        # Print summary for this model
        print(f"\nModel: {model_name}")
        print(f"Total responses: {total}")
        
        # Save results if requested
        if results_file:
            with open(results_file, 'a') as f:
                f.write(f"\nModel: {model_name}\n")
                f.write(f"Total responses: {total}\n")
        
        # Store counts for plotting
        model_response_counts[model_name] = counter
    
    # Get all unique responses across all models
    all_responses = set()
    for counter in model_response_counts.values():
        all_responses.update(counter.keys())
    
    # Sort responses by total count across all models
    response_total_counts = Counter()
    for counter in model_response_counts.values():
        response_total_counts.update(counter)
    
    top_responses = [resp for resp, _ in response_total_counts.most_common(top_n)]
    
    # Print comparison table
    print("\nResponse distribution across models:")
    header = "Response".ljust(30) + " | " + " | ".join(f"{model}".ljust(15) for model in model_response_counts.keys())
    print(header)
    print("-" * len(header))
    
    if results_file:
        with open(results_file, 'a') as f:
            f.write("\nResponse distribution across models:\n")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
    
    for response in top_responses:
        row = response[:28].ljust(30) + " | "
        for model_name, counter in model_response_counts.items():
            count = counter.get(response, 0)
            total = len(word_dict[model_name])
            percentage = (count / total) * 100 if total > 0 else 0
            cell = f"{count} ({percentage:.1f}%)".ljust(15)
            row += cell + " | "
        
        print(row)
        if results_file:
            with open(results_file, 'a') as f:
                f.write(row + "\n")
    
    # Create a single comparative bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up the bar positions
    bar_width = 0.8 / len(model_response_counts)
    index = np.arange(len(top_responses))
    
    # Create bars for each model
    for i, (model_name, counter) in enumerate(model_response_counts.items()):
        model_counts = []
        model_percentages = []
        
        for response in top_responses:
            count = counter.get(response, 0)
            total = len(word_dict[model_name])
            percentage = (count / total) * 100 if total > 0 else 0
            model_counts.append(count)
            model_percentages.append(percentage)
        
        # Plot percentages instead of raw counts for fair comparison
        position = index + (i - len(model_response_counts)/2 + 0.5) * bar_width
        ax.bar(position, model_percentages, bar_width, label=model_name, alpha=0.7)
    
    # Customize the plot
    ax.set_ylabel('Percentage of Responses (%)', fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xticks(index)
    ax.set_xticklabels(['\n'.join(wrap(resp, 20)) for resp in top_responses], rotation=45, ha='right', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filepath}_comparison.pdf", bbox_inches='tight', dpi=300)
    plt.close()


def process_numbers(words):
    """Extract numbers from a list of text responses."""
    numbers = []
    for word in words:
        word = word.strip()
        try:
            numbers.append(float(word))
        except ValueError:
            # Try to extract numbers from text
            number_matches = re.findall(r'\b\d+(?:\.\d+)?\b', word)
            if number_matches:
                try:
                    numbers.append(float(number_matches[0]))
                except ValueError:
                    pass
    return numbers


def numerical_bar_plot(question_name, title, word_dict, filepath, results_file=None, figsize=(12, 6)):
    """Create bar plots for numerical responses and print/save results."""
    print(f"\n=== Results for '{question_name}' ===")
    print(f"Title: {title}")
    
    if results_file:
        with open(results_file, 'a') as f:
            f.write(f"\n=== Results for '{question_name}' ===\n")
            f.write(f"Title: {title}\n")
    
    # Process all word lists
    model_stats = {}
    all_valid_numbers = {}
    
    for model_name, responses in word_dict.items():
        numbers = process_numbers(responses)
        all_valid_numbers[model_name] = numbers
        
        # Print results for this model
        print(f"\nModel: {model_name}")
        print(f"Total responses: {len(responses)}")
        print(f"Valid numerical responses: {len(numbers)}")
        
        if numbers:
            mean = np.mean(numbers)
            median = np.median(numbers)
            std_dev = np.std(numbers)
            min_val = min(numbers)
            max_val = max(numbers)
            
            # Calculate confidence intervals
            mean, ci = compute_confidence_intervals(numbers)
            
            model_stats[model_name] = {
                "mean": mean,
                "median": median,
                "std_dev": std_dev,
                "min": min_val,
                "max": max_val,
                "ci_low": ci[0],
                "ci_high": ci[1]
            }
            
            print(f"Mean: {mean:.2f}")
            print(f"Median: {median:.2f}")
            print(f"Min: {min_val:.2f}")
            print(f"Max: {max_val:.2f}")
            print(f"Standard deviation: {std_dev:.2f}")
            print(f"95% Confidence interval: [{ci[0]:.2f}, {ci[1]:.2f}]")
        
        # Save results if requested
        if results_file:
            with open(results_file, 'a') as f:
                f.write(f"\nModel: {model_name}\n")
                f.write(f"Total responses: {len(responses)}\n")
                f.write(f"Valid numerical responses: {len(numbers)}\n")
                
                if numbers:
                    f.write(f"Mean: {mean:.2f}\n")
                    f.write(f"Median: {median:.2f}\n")
                    f.write(f"Min: {min_val:.2f}\n")
                    f.write(f"Max: {max_val:.2f}\n")
                    f.write(f"Standard deviation: {std_dev:.2f}\n")
                    f.write(f"95% Confidence interval: [{ci[0]:.2f}, {ci[1]:.2f}]\n")
    
    # Print comparison table
    if model_stats:
        print("\nComparison of numerical results across models:")
        header = "Metric".ljust(20) + " | " + " | ".join(f"{model}".ljust(15) for model in model_stats.keys())
        print(header)
        print("-" * len(header))
        
        if results_file:
            with open(results_file, 'a') as f:
                f.write("\nComparison of numerical results across models:\n")
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
        
        for metric in ["mean", "median", "std_dev", "min", "max"]:
            metric_name = metric.capitalize().ljust(20)
            row = metric_name + " | "
            
            for model_name in model_stats:
                value = model_stats[model_name][metric]
                cell = f"{value:.2f}".ljust(15)
                row += cell + " | "
            
            print(row)
            if results_file:
                with open(results_file, 'a') as f:
                    f.write(row + "\n")
        
        # Add confidence intervals
        row = "95% CI".ljust(20) + " | "
        for model_name in model_stats:
            ci_low = model_stats[model_name]["ci_low"]
            ci_high = model_stats[model_name]["ci_high"]
            cell = f"[{ci_low:.2f}, {ci_high:.2f}]".ljust(15)
            row += cell + " | "
        
        print(row)
        if results_file:
            with open(results_file, 'a') as f:
                f.write(row + "\n")
    
    # Create a comparative bar plot for means with error bars
    if model_stats:
        fig, ax = plt.subplots(figsize=figsize)
        
        models = list(model_stats.keys())
        means = [model_stats[model]["mean"] for model in models]
        errors = [(model_stats[model]["mean"] - model_stats[model]["ci_low"], 
                  model_stats[model]["ci_high"] - model_stats[model]["mean"]) for model in models]
        
        x = np.arange(len(models))
        width = 0.6
        
        rects = ax.bar(x, means, width, yerr=np.transpose(errors),
                      align='center', alpha=0.7, ecolor='black', capsize=10)
        
        # Add value labels on top of each bar
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Add a horizontal line for the overall mean if there are multiple models
        if len(models) > 1:
            all_numbers = []
            for numbers in all_valid_numbers.values():
                all_numbers.extend(numbers)
            
            if all_numbers:
                overall_mean = np.mean(all_numbers)
                ax.axhline(y=overall_mean, color='r', linestyle='--', alpha=0.7)
                ax.annotate(f'Overall Mean: {overall_mean:.2f}',
                           xy=(0.5, overall_mean),
                           xytext=(0, 10),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           color='r')
        
        ax.set_ylabel('Value', fontsize=14)
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{filepath}_comparison.pdf", bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create a box plot to show distribution
        fig, ax = plt.subplots(figsize=figsize)
        
        box_data = [all_valid_numbers[model] for model in models]
        ax.boxplot(box_data, labels=models, showfliers=True, showmeans=True)
        
        ax.set_ylabel('Value', fontsize=14)
        ax.set_title(f"{title} - Distribution", fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{filepath}_boxplot.pdf", bbox_inches='tight', dpi=300)
        plt.close()


def extract_responses_by_question(chat_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract responses from chat data, grouped by question name and model.
    
    Args:
        chat_data: List of chat items with messages and metadata
        
    Returns:
        Dictionary mapping question names to dictionaries mapping model names to lists of responses
    """
    responses_by_question = {}
    
    for item in chat_data:
        if "metadata" not in item or "name" not in item["metadata"]:
            continue
            
        question_name = item["metadata"]["name"]
        
        # Extract model name if available, otherwise use "unknown"
        model_name = item.get("model", "unknown")
        
        # Get the assistant's response
        if len(item["messages"]) >= 2 and item["messages"][1]["role"] == "assistant":
            response = item["messages"][1]["content"].strip()
            
            # Initialize nested dictionaries if needed
            if question_name not in responses_by_question:
                responses_by_question[question_name] = {}
            if model_name not in responses_by_question[question_name]:
                responses_by_question[question_name][model_name] = []
                
            # Add the response
            responses_by_question[question_name][model_name].append(response)
    
    return responses_by_question


def main():
    parser = argparse.ArgumentParser(description='Plot responses from chat format JSON')
    parser.add_argument('--input_files', type=str, nargs='+', required=True,
                        help='Input JSON file(s) with chat format responses')
    parser.add_argument('--output_dir', '--ouput_dir', type=str, default='plots',
                        help='Directory to save plots')
    parser.add_argument('--plot_type', type=str, choices=['text', 'number'], default='text',
                        help='Type of plot to generate (text or number)')
    parser.add_argument('--results_file', type=str, default=None,
                        help='File to save key results (optional)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clear results file if specified
    if args.results_file:
        with open(args.results_file, 'w') as f:
            f.write(f"Results generated from {os.path.basename(__file__)}\n")
            f.write(f"Input files: {', '.join(args.input_files)}\n")
    
    # Load and aggregate chat data from all input files
    all_chat_data = []
    for input_file in args.input_files:
        try:
            # Handle both relative and absolute paths
            if os.path.isabs(input_file):
                file_path = input_file
            else:
                file_path = os.path.join(os.getcwd(), input_file)
                
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            with open(file_path, 'r') as f:
                chat_data = json.load(f)
                all_chat_data.extend(chat_data)
                print(f"Loaded {len(chat_data)} items from {input_file}")
        except FileNotFoundError:
            print(f"Warning: File not found: {input_file}")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in file: {input_file}")
        except Exception as e:
            print(f"Error processing file {input_file}: {str(e)}")
    
    if not all_chat_data:
        print("Error: No valid data loaded from input files.")
        return
    
    # Extract responses by question
    responses_by_question = extract_responses_by_question(all_chat_data)
    
    if not responses_by_question:
        print("Error: No valid responses found in the data.")
        return
    
    # Print summary of the data
    print("\n=== Data Summary ===")
    print(f"Total items loaded: {len(all_chat_data)}")
    print(f"Questions found: {len(responses_by_question)}")
    
    # Count total responses by model
    model_counts = {}
    for question_data in responses_by_question.values():
        for model_name, responses in question_data.items():
            if model_name not in model_counts:
                model_counts[model_name] = 0
            model_counts[model_name] += len(responses)
    
    print("Responses by model:")
    for model_name, count in model_counts.items():
        print(f"  - {model_name}: {count} responses")
    
    print("\nQuestions with responses:")
    for question_name, model_data in responses_by_question.items():
        total_responses = sum(len(responses) for responses in model_data.values())
        print(f"  - {question_name}: {total_responses} responses")
    
    # Aggregate all responses by model (across all questions)
    if args.plot_type == 'text':
        # For text responses, we'll create a single plot showing the most common responses across all questions
        all_responses_by_model = {}
        
        for model_name in model_counts.keys():
            all_responses_by_model[model_name] = []
            
            # Collect all responses for this model across all questions
            for question_data in responses_by_question.values():
                if model_name in question_data:
                    # Filter out numeric responses
                    non_numeric_responses = [resp for resp in question_data[model_name] 
                                           if not resp.replace('.', '', 1).isdigit()]
                    all_responses_by_model[model_name].extend(non_numeric_responses)
        
        print("\n=== Aggregate Results Across All Questions ===")
        filepath = os.path.join(args.output_dir, "aggregate_text")
        free_form_bar_plot("aggregate", "Aggregate Responses Across All Questions", 
                          all_responses_by_model, filepath, args.results_file)
    else:  # number
        # For numerical responses, we'll aggregate the values directly without normalization
        
        # Get all model names
        all_models = set()
        for question_data in responses_by_question.values():
            all_models.update(question_data.keys())
        
        # Get questions with numerical responses
        numerical_questions = []
        question_titles = {}
        
        for question_name, model_responses in responses_by_question.items():
            # Check if this question has numerical responses
            has_numbers = False
            for responses in model_responses.values():
                numbers = process_numbers(responses)
                if numbers:
                    has_numbers = True
                    break
            
            if has_numbers:
                numerical_questions.append(question_name)
                
                # Get the title for this question
                title = question_name
                for item in all_chat_data:
                    if "metadata" in item and item["metadata"].get("name") == question_name:
                        title = item["metadata"].get("title", question_name)
                        break
                
                question_titles[question_name] = title
        
        if not numerical_questions:
            print("Error: No questions with valid numerical responses found.")
            return
        
        print(f"\n=== Aggregate Results for {len(numerical_questions)} Questions with Numerical Responses ===")
        
        # Collect all numerical data across all questions
        all_values_by_model = {model: [] for model in all_models}
        question_values_by_model = {model: {} for model in all_models}
        
        # Process each question
        for question_name in numerical_questions:
            model_responses = responses_by_question[question_name]
            title = question_titles[question_name]
            
            # Process numerical responses for this question
            all_valid_numbers = {}
            
            for model_name, responses in model_responses.items():
                numbers = process_numbers(responses)
                if numbers:
                    all_valid_numbers[model_name] = numbers
                    # Add raw values to the collection
                    all_values_by_model[model_name].extend(numbers)
                    question_values_by_model[model_name][question_name] = numbers
            
        # Print aggregate statistics
        print("\n=== Aggregate Statistics (Raw Values) ===")
        
        aggregate_stats = {}
        for model_name, values in all_values_by_model.items():
            if values:
                mean = np.mean(values)
                median = np.median(values)
                std_dev = np.std(values)
                min_val = min(values)
                max_val = max(values)
                
                # Calculate confidence intervals
                mean, ci = compute_confidence_intervals(values)
                
                aggregate_stats[model_name] = {
                    "mean": mean,
                    "median": median,
                    "std_dev": std_dev,
                    "min": min_val,
                    "max": max_val,
                    "ci_low": ci[0],
                    "ci_high": ci[1]
                }
        
        # Print comparison table for aggregate stats
        if aggregate_stats:
            header = "Metric".ljust(20) + " | " + " | ".join(f"{model}".ljust(15) for model in aggregate_stats.keys())
            print(header)
            print("-" * len(header))
            
            if args.results_file:
                with open(args.results_file, 'a') as f:
                    f.write("\n=== Aggregate Statistics (Raw Values) ===\n")
                    f.write(header + "\n")
                    f.write("-" * len(header) + "\n")
            
            for metric in ["mean", "median", "std_dev", "min", "max"]:
                metric_name = metric.capitalize().ljust(20)
                row = metric_name + " | "
                
                for model_name in aggregate_stats:
                    value = aggregate_stats[model_name][metric]
                    cell = f"{value:.2f}".ljust(15)
                    row += cell + " | "
                
                print(row)
                if args.results_file:
                    with open(args.results_file, 'a') as f:
                        f.write(row + "\n")
        
        # Create a single bar plot for the aggregate values
        if aggregate_stats:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            models = list(aggregate_stats.keys())
            means = [aggregate_stats[model]["mean"] for model in models]
            errors = [(aggregate_stats[model]["mean"] - aggregate_stats[model]["ci_low"], 
                      aggregate_stats[model]["ci_high"] - aggregate_stats[model]["mean"]) for model in models]
            
            x = np.arange(len(models))
            width = 0.6
            
            rects = ax.bar(x, means, width, yerr=np.transpose(errors),
                          align='center', alpha=0.7, ecolor='black', capsize=10)
            
            # Add value labels on top of each bar
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            ax.set_ylabel('Value', fontsize=14)
            ax.set_title('Aggregate Comparison Across All Questions', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "aggregate_raw.pdf"), bbox_inches='tight', dpi=300)
            plt.close()
        
        # Create a box plot for the raw values
        if all_values_by_model:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            models = [model for model, values in all_values_by_model.items() if values]
            box_data = [all_values_by_model[model] for model in models]
            
            ax.boxplot(box_data, labels=models, showfliers=True, showmeans=True)
            ax.set_ylabel('Value', fontsize=14)
            ax.set_title('Distribution of Values Across All Questions', fontsize=16)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "aggregate_raw_boxplot.pdf"), bbox_inches='tight', dpi=300)
            plt.close()


if __name__ == "__main__":
    main() 