import json


def print_hard_labels_from_first_file(file_path_1, file_path_2):
    with open(file_path_1, 'r', encoding='utf-8') as file1, open(file_path_2, 'r', encoding='utf-8') as file2:
        second_file_data = [json.loads(line) for line in file2]

        data_point_number = 1  # Initialize the counter for data points

        for line1, line2 in zip(file1, second_file_data):
            try:
                # Parse first file line as JSON
                item1 = json.loads(line1)
                text1 = item1["model_output_text"]
                text2 = item1["model_input"]
                hard_labels_1 = item1["hard_labels"]

                # Parse second file line as JSON
                modeltext = line2["model_output_text"]
                hard_labels_2 = line2["hard_labels"]


                # Print the data point number
                print(f"DATAPOINT {data_point_number}")
                print(f"Input: {text2}")
                print(f"Output: {text1}")


                # Print the text slices for both files
                for start, end in hard_labels_1:
                    print(f"TEST Spans: {text1[start:end]}")

                print(f"FLAN-T5 span: {modeltext}")
                for start, end in hard_labels_2:
                    print(f"FENJI span: {text1[start:end]}")

                # Separator between different text comparisons
                print("-" * 40)

                # Increment the data point number
                data_point_number += 1

            except json.JSONDecodeError:
                print(f"Error decoding JSON on one of the lines.")
                continue


# Example usage
file_path_1 = "mushroom.en-test.v1.jsonl"  # Path to the labeled TEST JSON file
file_path_2 = "mushroom.labels.en.tst.v1.dprplus.jsonl"  # Path to FENJI TEST JSON file

# Print hard labels and corresponding text from both files for comparison
print_hard_labels_from_first_file(file_path_1, file_path_2)
