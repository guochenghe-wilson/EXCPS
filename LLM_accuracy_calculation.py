import os
import json
import openai 
import time

from openai import OpenAI

def connect_gpt_4o(input):
  client = OpenAI(api_key="your_api_key")

  response = client.responses.create(
    model="gpt-4o",
    input=[
      {
        "role": "system",
        "content": [
          {
            "type": "input_text",
            "text": "Given a dimension (either x or y), a key feature description, and trajectory data, calculate the associated quantitative velocity value along the specified dimension that is most relevant to the key feature.\n\nData Description: The trajectory consists of sequential (x, y) coordinates sampled at 0.1-second intervals.\n\nCalculation Instructions:\nIdentify the relevant timestamp T from the key feature.\nThen, extract the value at indices T - 2 and T along the specified dimension.\nCompute the absolute difference between these two values, and divide it by 2 × time interval (i.e., 0.2 seconds) to obtain the velocity.\n\nExample 1: \n{\"dimension\": y, \"key_feature\": \"observed position of agent_4 at scene time 20 at the 19 timestamp\",\"extracted_info\": [[26.47,41.25],[26.37,40.98],[26.19,40.67],[26.05,40.37],[25.83,39.95],[25.59,39.43],[25.28,38.85],[25.09,38.51],[25.1,38.46],[24.94,38.08],[24.71,37.68],[24.54,37.33],[24.4,37.0],[24.22,36.71],[23.97,36.28],[23.78,35.89],[23.59,35.5],[23.3,34.95],[23.09,34.56],[22.85,34.12]]}\ncalculation: absolute value of {(34.12-34.95)/0.2}\noutput = 4.15\n\nExample 2:\n{\"dimension\": x, \"key_feature\": \"observed position of agent_4 at scene time 20 at the 19 timestamp\",\"extracted_info\": [[26.47,41.25],[26.37,40.98],[26.19,40.67],[26.05,40.37],[25.83,39.95],[25.59,39.43],[25.28,38.85],[25.09,38.51],[25.1,38.46],[24.94,38.08],[24.71,37.68],[24.54,37.33],[24.4,37.0],[24.22,36.71],[23.97,36.28],[23.78,35.89],[23.59,35.5],[23.3,34.95],[23.09,34.56],[22.85,34.12]]}\ncalculation: absolute value of {(22.85-23.3)/0.2}\noutput = 0.75\n\nImportant: Output only the numerical value of the calculated velocity.\nDo not include any units, explanations, or additional text."
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": input
          }
        ]
      }
    ],
    text={
      "format": {
        "type": "text"
      }
    },
    reasoning={},
    tools=[],
    temperature=1,
    max_output_tokens=2048,
    top_p=1,
    store=True
  )

  gpt_response_message_content = next((out.content[0].text for out in response.output if out.type == 'message'), None)

  return gpt_response_message_content


def connect_gpt_o1(input):
  client = OpenAI(api_key="your_api_key")

  response = client.responses.create(model="o1", input=[
    {
      "role": "developer",
      "content": [
        {
          "type": "input_text",
          "text": "Given a dimension (either x or y), a key feature description, and trajectory data, calculate the associated quantitative velocity value along the specified dimension that is most relevant to the key feature.\n\nData Description: The trajectory consists of sequential (x, y) coordinates sampled at 0.1-second intervals.\n\nCalculation Instructions:\nIdentify the relevant timestamp T from the key feature.\nThen, extract the value at indices T - 2 and T along the specified dimension.\nCompute the absolute difference between these two values, and divide it by 2 × time interval (i.e., 0.2 seconds) to obtain the velocity.\n\nExample 1: \n{\"dimension\": y, \"key_feature\": \"observed position of agent_4 at scene time 20 at the 19 timestamp\",\"extracted_info\": [[26.47,41.25],[26.37,40.98],[26.19,40.67],[26.05,40.37],[25.83,39.95],[25.59,39.43],[25.28,38.85],[25.09,38.51],[25.1,38.46],[24.94,38.08],[24.71,37.68],[24.54,37.33],[24.4,37.0],[24.22,36.71],[23.97,36.28],[23.78,35.89],[23.59,35.5],[23.3,34.95],[23.09,34.56],[22.85,34.12]]}\ncalculation: absolute value of {(34.12-34.95)/0.2}\noutput = 4.15\n\nExample 2:\n{\"dimension\": x, \"key_feature\": \"observed position of agent_4 at scene time 20 at the 19 timestamp\",\"extracted_info\": [[26.47,41.25],[26.37,40.98],[26.19,40.67],[26.05,40.37],[25.83,39.95],[25.59,39.43],[25.28,38.85],[25.09,38.51],[25.1,38.46],[24.94,38.08],[24.71,37.68],[24.54,37.33],[24.4,37.0],[24.22,36.71],[23.97,36.28],[23.78,35.89],[23.59,35.5],[23.3,34.95],[23.09,34.56],[22.85,34.12]]}\ncalculation: absolute value of {(22.85-23.3)/0.2}\noutput = 0.75\n\nImportant: Output only the numerical value of the calculated velocity.\nDo not include any units, explanations, or additional text."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": input
        }
      ]
    }
  ],
    text={
    "format": {
      "type": "text"
    }
  },
    reasoning={
    "effort": "low"
  },
    tools=[],
    store=True
  )

  gpt_response_message_content = next((out.content[0].text for out in response.output if out.type == 'message'), None)

  return gpt_response_message_content
    

def evaluate_4o_assistant_accuracy(folder_path):
    correct = 0
    total = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                dimension = data.get("dimension")
                key_feature = data.get("key_feature")
                extracted_info = data.get("extracted_info")
                ground_truth = data.get("ground_truth")

                query=dimension + " " + key_feature + " " + str(extracted_info)
                # print(query)

                if key_feature and extracted_info and ground_truth is not None:
                    assistant_value = connect_gpt_4o(query)
                    if abs(abs(float(assistant_value)) - abs(ground_truth)) <= 0.01:
                        correct += 1
                    total += 1

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    if total == 0:
        print("No valid files processed.")
        return 0.0

    accuracy = (correct / total) * 100
    print(f"Assistant Accuracy: {correct}/{total} = {accuracy:.2f}%")
    return accuracy

def evaluate_o1_assistant_accuracy(folder_path):
    correct = 0
    total = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                dimension = data.get("dimension")
                key_feature = data.get("key_feature")
                extracted_info = data.get("extracted_info")
                ground_truth = data.get("ground_truth")

                query =dimension + " " + key_feature + " " + str(extracted_info)

                if key_feature and extracted_info and ground_truth is not None:
                    assistant_value = connect_gpt_o1(query)
                    if abs(abs(float(assistant_value)) - abs(ground_truth)) <= 0.01:
                        correct += 1
                    total += 1

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    if total == 0:
        print("No valid files processed.")
        return 0.0

    accuracy = (correct / total) * 100
    print(f"Assistant Accuracy: {correct}/{total} = {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":

    folder_path_ref = "your_folder_path"
    folder_path_vinilla = "your_folder_path"
    accuracy_4o = evaluate_4o_assistant_accuracy(folder_path_ref, folder_path_vinilla)
    accuracy_o1 = evaluate_o1_assistant_accuracy(folder_path_test)
    print(f"The accuracy of quantitative value calculation after the logical reasoning is: {accuracy_4o:.2f}%")
    print(f"The accuracy of quantitative value calculation after the logical reasoning is: {accuracy_o1:.2f}%")

    folder_path_ref = "your_folder_path"
    folder_path_vinilla = "your_folder_path"
    accuracy_4o = evaluate_4o_assistant_accuracy(folder_path_ref, folder_path_vinilla)
