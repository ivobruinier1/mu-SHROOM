import json
import transformers
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
def read_data(set, passages):
    """ This function
    1. reads the .jsonl validation or test set and retrieves
    the model input and the model output text
    2. reads the passsages retrieved from the DPR system"""

    pairs_to_test = []
    # Read data from sample_test.v1.json
    with open(set, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            # Collect id, model_output_text, model_input, and soft_labels as pairs
            pairs_to_test.append((data['id'], data['model_output_text'], data['model_input'], data['soft_labels']))


    # Dictionary to store the IDs and top passages
    stored_data = {}

    with open(passages, "r") as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            if key.startswith("val-en"):
                results = value.get("results", [])
                top_passage = results[0][0]
                stored_data[key] = top_passage

    appended_pairs = []
    for pair in pairs_to_test:
        pair_id = pair[0]
        top_passage = stored_data.get(pair_id, None)  # Retrieve passage or use None if not found
        appended_pairs.append((*pair, top_passage))

    print("Printing first 3 datapoints...")
    for pair in appended_pairs[0:3]:
        print(pair)  # debug check
    return appended_pairs


def make_prompts(appended_pairs):
    """ This function
     1. retrieves the model input storen and coherent passage
     2. Formats a prompt with the given data"""

    prompts = {}
    for pair in appended_pairs:
        id = pair[0]
        hypothesis = pair[1]
        question = pair[2]
        context = pair[4]

        prompt = (f" Question = {question}\n Hypothesis {hypothesis}\n Using the provided context, identify the part of the hypothesis that contradicts the premise.\n context = {context}")
        prompts[id] = str(prompt)
    print("Printing prompts...")
    print(prompts)
    return prompts


def prompt_model(prompt):
    """ This function
    1. Uses the given t5 model to prompt it with the context gained from the
     retrieved passages and the model input and output"""

    model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    input = prompt
    encoded_input = tokenizer([input],
                                return_tensors='pt',
                                max_length=512,
                                truncation=True)
    output = model.generate(input_ids=encoded_input.input_ids,
                            attention_mask=encoded_input.attention_mask)
    model_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return str(model_output)



def stash_prompt(appended_pairs, prompt):
    for pair in appended_pairs:
        appended_pairs.append(prompt)

def get_overlap_span(text1, text2):
    """ # Example usage
input1 = "Petra van Stoveren won a silver medal in the 2008 summer Olympics in Beijing, China."
input2 = "2008"

result = get_overlap_span(input1, input2)
print(result)
"""

    # lowercasing both texts
    text1 = text1.lower()
    text2 = text2.lower()

    start = text1.find(text2)
    if start == -1:
        return None
    end = start + len(text2) - 1
    return [f"start:{start}, end:{end}"]


if __name__ == "__main__":
    print("Running script...")
    set = "mushroom.en-val.v2.jsonl"
    passages = "retrieved_passages.json"
    appended_pairs = read_data(set, passages)
    prompts = make_prompts(appended_pairs)
    print("Prompting model...")
    halu_text_spans = []
    for prompt in prompts.values():
        output = prompt_model(prompt)
        halu_text_spans.append(output)
    updated_data = []
    # Loop through each data point in the tuple and corresponding item in the list
    for data_point, extra_item in zip(appended_pairs, halu_text_spans):
        updated_data.append(data_point + (extra_item,))
    hard_spans = []
    for pairs in updated_data:
        hard_span = get_overlap_span(pairs[1], pairs[5])
        hard_spans.append(hard_span)
    finished_data = []
    for data_point, extra_item in zip(updated_data, hard_spans):
        finished_data.append(data_point + (extra_item,))
    print(hard_spans)
    output_file = "hard_spans.json"
    with open(output_file, "w") as f:
        json.dump(finished_data, f, indent=3)













