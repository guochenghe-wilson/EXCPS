import os
import json
import numpy as np

def compute_correctness_percentage(folder_path):
    correct_count = 0
    total_count = 0

    # Loop over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                q_val = data.get("quantitative_value")
                gt_val = data.get("ground_truth")

                if q_val is not None and gt_val is not None:
                    total_count += 1
                    if np.abs(np.abs(q_val) - np.abs(gt_val)) < 0.01:
                        correct_count += 1

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Calculate and return percentage
    if total_count == 0:
        print("No valid JSON files with quantitative_value and ground_truth found.")
        return 0.0

    percentage = (correct_count / total_count) * 100
    print(f"Correct: {correct_count}/{total_count} ({percentage:.2f}%)")
    return percentage

if __name__ == "__main__":
    folder_path = "/Users/guocheng/Dropbox/EX-CPS/LLMxAIchatBot/evaluate/output_jsons_after_logical_calculation"
    accuracy = compute_correctness_percentage(folder_path)
    print(f"The accuracy of quantitative value calculation after the logical reasoning is: {accuracy:.2f}%")
