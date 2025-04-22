import json
import re
import os

def update_json_with_delta(file_path):

    filename = os.path.basename(file_path)

    # Extract T and dim from filename using regex
    match = re.search(r'_T\d+_F\d+_S\d+', filename)
    if not match:
        raise ValueError("Filename does not match expected pattern '_T## _F## _S##'")

    # Extract values
    T_match = re.search(r'S(\d+)', filename)
    dim_match = re.search(r'F(\d+)', filename)

    if not T_match or not dim_match:
        raise ValueError("Could not extract T or F from filename")

    T = int(T_match.group(1)) - 1  
    dim = int(dim_match.group(1))

    if dim not in [20, 21]:
        raise ValueError("dim must be 20 (x) or 21 (y)")

    with open(file_path, 'r') as f:
        data = json.load(f)

    trajectory = data["extracted_info"]

    if not (1 <= T < len(trajectory)):
        raise IndexError(f"T={T} is out of range for trajectory of length {len(trajectory)}")

    # Calculate delta
    prev_coord = trajectory[T - 1]
    curr_coord = trajectory[T + 1]
    idx = 0 if dim == 20 else 1
    delta = (curr_coord[idx] - prev_coord[idx]) / 0.2

    # Update field
    data["quantitative_value"] = delta

    # Save updated file
    with open(file_path, 'w') as f:
        json.dump(data, f, separators=(',', ': '))

    print(f"Updated delta ({'x' if dim == 20 else 'y'}) = {delta:.6f} saved in 'quantitative_value' of {file_path}")

if __name__ == "__main__":
    update_json_with_delta("/Users/guocheng/Dropbox/EX-CPS/LLMxAIchatBot/evaluate/output_jsons/scene_42_T23_F20_S19.json")

