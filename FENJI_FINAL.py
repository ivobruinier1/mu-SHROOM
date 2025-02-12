import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelWithLMHead
import re
import string
from difflib import SequenceMatcher

def read_data(set_file, passages_file):
    """
    Reads data from a .jsonl file and retrieves model input/output and associated passages.
    """
    # Read pairs to test
    pairs_to_test = []
    with open(set_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            pairs_to_test.append((data['id'], data['model_output_text'], data['model_input']))

    # Read passages retrieved by the DPR system
    stored_data = {}
    with open(passages_file, "r") as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            if key.startswith("tst"):
                results = value.get("results", [])
                top_passage = results[0][0:2] if results else None
                stored_data[key] = top_passage

    # Combine pairs with their corresponding passages
    appended_pairs = [
        (*pair, stored_data.get(pair[0], None)) for pair in pairs_to_test
    ]

    # Debug: Print the first 3 data points
    print("\nPrinting first 3 datapoints for debugging...\n")
    for pair in appended_pairs[:3]:
        print(pair)

    return appended_pairs


def make_prompts(appended_pairs):
    """
    Formats prompts using the given data, including context and hypotheses.
    """
    prompts = {}
    for pair in appended_pairs:
        pair_id, hypothesis, question, context = pair
        prompt = (
            f"Question = {question}. Hypothesis = {hypothesis}. "
            f"Using this context: {context}, identify and cite the EXACT part of the hypothesis that "
            f"contradicts the premise by giving the textual span, make sure to not add ANY other words. "
        )
        prompts[pair_id] = prompt
    return prompts


def prompt_model(prompt, model):
    """
    Uses the T5 model to generate output based on the given prompt.
    """
    model_name = model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)

    encoded_input = tokenizer([prompt], return_tensors='pt', max_length=512, truncation=True)
    output = model.generate(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def get_overlap_span(text1, text2):
    """
    Finds the overlap span between two texts with enhanced matching.
    """

    def preprocess_text(text):
        # Normalize text: lowercase, remove punctuation, and extra spaces
        text = ''.join(char for char in text if char not in string.punctuation)
        return ' '.join(text.lower().split())  # Remove extra spaces

    if text2 != "None":    # Preprocess texts
        text1, text2 = preprocess_text(text1), preprocess_text(text2)

        # Use difflib to find the best match
        matcher = SequenceMatcher(None, text1, text2)
        match = matcher.find_longest_match(0, len(text1), 0, len(text2))

        if match.size > 0:
            start, end = match.a, match.a + match.size - 1
            if end == start:
                end += 1
            return {"start": start, "end": end}
        else:
            return None
    else:
        return None


def reformat(original_data):
    """
    Reformats the data into a standardized structure with soft and hard labels.
    """
    output_records = []
    for entry in original_data:
        record_id, possible_hallucination, _, soft_labels_entry, model_output_text, hard_label_entry = entry

        # Format hard labels
        hard_labels = [[hard_label_entry["start"], hard_label_entry["end"]]] if isinstance(hard_label_entry, dict) else []

        # Format soft labels
        soft_labels = []
        max_end = 0

        output_records.append({
            "id": record_id,
            "soft_labels": soft_labels,
            "hard_labels": hard_labels,
            "possible_hallucination": possible_hallucination,
            "model_output_text": model_output_text,
        })
    return output_records


def count_entries_with_hard_labels(data):
    entries = 0
    labels = 0
    count = 0

    for entry in data:
        entries += 1
        if entry[5] != None:
            labels += 1
        if entry[4] != "None":
            count += 1
    return labels, entries, count




if __name__ == "__main__":

    languages = ["ar", "ca", "cs", "de", "en", "es", "eu", "fa", "fi", "fr", "hi", "it", "sv", "zh"]
    # Here you can adjust which languages you want to scan for hallucinations

    for language in languages:
        print(f"\nRunning script for the language: {language}\n")
        # Load data
        set_file = f"test_set/mushroom.{language}-tst.v1.jsonl"# These are the file(s) you use for testing
        passages_file = f"passages_v2/retrieved_passages_{language}_v2.json"  # Here you can enter the DPR File(s) that are !coherent! to the set_file
        model = "google/flan-t5-base" # Here you can change the model you want to experiment with
        appended_pairs = read_data(set_file, passages_file)

        # Generate prompts
        prompts = make_prompts(appended_pairs)
        print("\nPrompting model...\n")

        # Get hallucinated text spans
        halu_text_spans = [prompt_model(prompt, model) for prompt in prompts.values()]

        # Update and process data
        updated_data = [data_point + (span,) for data_point, span in zip(appended_pairs, halu_text_spans)]
        hard_spans = [get_overlap_span(data[1], data[4]) for data in updated_data]
        finished_data = [data_point + (span,) for data_point, span in zip(updated_data, hard_spans)]

        # Reformat and save output
        hard_labels = reformat(finished_data)

        labels, entries, count = count_entries_with_hard_labels(finished_data)
        print(f"\nThere have been {count} hallucinations out of {entries} {language} data entries found!\n")
        print(f"\n{labels} hard labels were successfully extracted out of {count} hallucinations.")

        with open(f"mushroom.labels.{language}.tst.jsonl", "w") as f:
            f.write('\n'.join(json.dumps(record) for record in hard_labels))
