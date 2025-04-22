feature_groups = {
    "position_information":  [0, 1, 5, 6, 10, 11, 15, 16, 20, 21],
    "heading_information":   [2, 7, 12, 17, 22],
    "velocity_information":  [3, 4, 8, 9, 13, 14, 18, 19, 23, 24],
    "traffic_light_information": [25],
}

agent_groups = {
    "ego":        [0, 1, 2, 3, 4],
    "agent_1":    [5, 6, 7, 8, 9],
    "agent_2":    [10, 11, 12, 13, 14],
    "agent_3":    [15, 16, 17, 18, 19],
    "agent_4":    [20, 21, 22, 23, 24],
    "traffic_light": [25],
}

signal_key = {
    "position_information":  "observed_agent_trajectory",
    "heading_information":   "observed_agent_heading",
    "velocity_information":  "observed_agent_velocity",
    "traffic_light_information": "traffic_light_state"
}

import json
import os

# Agent mapping
agent_groups = {
    "ego": [0, 1, 2, 3, 4],
    "agent_1": [5, 6, 7, 8, 9], 
    "agent_2": [10, 11, 12, 13, 14],
    "agent_3": [15, 16, 17, 18, 19],
    "agent_4": [20, 21, 22, 23, 24],
    "traffic_light": [25]
}

def get_agent_group(feature_index):
    for agent, features in agent_groups.items():
        if feature_index in features:
            return agent
    raise ValueError(f"Feature {feature_index} not mapped to any agent.")

class CompactEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, dict) and "extracted_info" in obj:
            compact = super().encode(obj["extracted_info"])
            obj_copy = obj.copy()
            obj_copy["extracted_info"] = "<<EXTRACTED_INFO>>"
            json_str = super().encode(obj_copy)
            return json_str.replace('"<<EXTRACTED_INFO>>"', compact)
        return super().encode(obj)

def extract_and_save_formatted(scene_path, time_str, feature_index, step, shap_score=0.0, output_dir="./output_jsons"):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(scene_path, 'r') as f:
        data = json.load(f)

    agent_group = get_agent_group(feature_index)
    agent_key = agent_group if agent_group == "traffic_light" else f"agent_{agent_group.split('_')[-1]}"

    try:
        trajectory_data = data["scene_data"]["time_stamped_data"]["trajectory_module"]["observed_agents"]["front_camera"]
        agent_data = trajectory_data[agent_key][str(time_str)]["input"]["observed_agent_trajectory"]
    except KeyError as e:
        raise KeyError(f"Missing expected key in JSON structure: {e}")

    output_dict = {
        "key_feature": f"observed position of {agent_group} at scene time {time_str} at the {step} timestamp",
        "extracted_info": agent_data,
        "quantitative_value": shap_score,
        "ground_truth": 0
    }

    output_filename = f"{os.path.splitext(os.path.basename(scene_path))[0]}_T{time_str}_F{feature_index}_S{step}.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as out_f:
        out_f.write(json.dumps(output_dict, cls=CompactEncoder))

    print(f"Compact-formatted output saved to {output_path}")

if __name__ == "__main__":

    extract_and_save_formatted(
        scene_path="/Users/guocheng/Dropbox/EX-CPS/LLMxAIchatBot/data/planner_training/processed_data/scene_12.json",
        time_str="117",
        feature_index=21,
        step=19,
        shap_score=7.207413
    )
