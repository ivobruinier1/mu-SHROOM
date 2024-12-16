import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelWithLMHead


def read_data(set_file, passages_file):
    """
    Reads data from a .jsonl file and retrieves model input/output and associated passages.
    """
    # Read pairs to test
    pairs_to_test = []
    with open(set_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            pairs_to_test.append((data['id'], data['model_output_text'], data['model_input'], data['soft_labels']))

    # Read passages retrieved by the DPR system
    stored_data = {}
    with open(passages_file, "r") as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            if key.startswith("val-en"):
                results = value.get("results", [])
                top_passage = results[0][0:2] if results else None
                stored_data[key] = top_passage

    # Combine pairs with their corresponding passages
    appended_pairs = [
        (*pair, stored_data.get(pair[0], None)) for pair in pairs_to_test
    ]

    # Debug: Print the first 3 data points
    print("Printing first 3 datapoints...")
    for pair in appended_pairs[:3]:
        print(pair)

    return appended_pairs


def make_prompts(appended_pairs):
    """
    Formats prompts using the given data, including context and hypotheses.
    """
    prompts = {}
    for pair in appended_pairs:
        pair_id, hypothesis, question, _, context = pair
        prompt = (
            f"Question = {question}. Hypothesis = {hypothesis}. "
            f"Using the provided context, identify and cite the EXACT part of the hypothesis that "
            f"contradicts the premise by giving the textual span, make sure to not add ANY other words. "
            f"context = {context}"
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
    Finds the overlap span between two texts.
    """
    text1, text2 = text1.lower(), text2.lower()
    start = text1.find(text2)
    if start == -1:
        return None
    return {"start": start, "end": start + len(text2) - 1}


def reformat(original_data):
    """
    Reformats the data into a standardized structure with soft and hard labels.
    """
    output_records = []
    for entry in original_data:
        record_id, _, _, soft_labels_entry, _, _, hard_label_entry = entry

        # Format hard labels
        hard_labels = [[hard_label_entry["start"], hard_label_entry["end"]]] if isinstance(hard_label_entry, dict) else []

        # Format soft labels
        soft_labels = []
        max_end = 0
        if isinstance(soft_labels_entry, list):
            for label in soft_labels_entry:
                if {"start", "end", "prob"}.issubset(label):
                    soft_labels.append(label)
                    max_end = max(max_end, label["end"])

        # Ensure model output text is long enough
        model_output_text = "x" * (max_end + 10)

        output_records.append({
            "id": record_id,
            "soft_labels": soft_labels,
            "hard_labels": hard_labels,
            "model_output_text": model_output_text,
        })
    return output_records


def recompute_hard_labels(soft_labels):
    """
    Infers hard labels from soft labels with a probability threshold.
    """
    hard_labels = []
    prev_end = -1
    for start, end in (
        (lbl['start'], lbl['end'])
        for lbl in sorted(soft_labels, key=lambda span: (span['start'], span['end']))
        if lbl['prob'] > 0.5
    ):
        if start == prev_end:
            hard_labels[-1][-1] = end
        else:
            hard_labels.append([start, end])
        prev_end = end
    return hard_labels


def load_jsonl_file_to_records(filename):
    """
    Reads data from a JSONL file into a DataFrame, computes hard labels if missing.
    """
    df = pd.read_json(filename, lines=True)
    if 'hard_labels' not in df.columns:
        df['hard_labels'] = df.soft_labels.apply(recompute_hard_labels)
    df['text_len'] = df.model_output_text.str.len()
    return df[['id', 'soft_labels', 'hard_labels', 'text_len']].sort_values('id').to_dict(orient='records')


def score_iou(ref_dict, pred_dict):
    """
    Computes IoU between reference and predicted hard labels for a single datapoint.
    """
    assert ref_dict['id'] == pred_dict['id']
    ref_indices = {idx for span in ref_dict['hard_labels'] for idx in range(*span)}
    pred_indices = {idx for span in pred_dict['hard_labels'] for idx in range(*span)}
    if not ref_indices and not pred_indices:
        return 1.0
    return len(ref_indices & pred_indices) / len(ref_indices | pred_indices)


def scorer(ref_dicts, pred_dicts):
    """
    Computes average IoU across all datapoints.
    """
    assert len(ref_dicts) == len(pred_dicts)
    return np.mean([score_iou(r, d) for r, d in zip(ref_dicts, pred_dicts)])


if __name__ == "__main__":
    print("Running script...")

    # Load data
    set_file = "mushroom.en-val.v2.jsonl" # This is the file you use for testing
    passages_file = "retrieved_passages.json"  # Here you can enter the DPR File that is !coherent! to the set_file
    model = "google/flan-t5-base" # Here you can change the model you want to experiment with
    appended_pairs = read_data(set_file, passages_file)

    # Generate prompts
    prompts = make_prompts(appended_pairs)
    print("Prompting model...")

    # Get hallucinated text spans
    halu_text_spans = [prompt_model(prompt, model) for prompt in prompts.values()]

    # Update and process data
    updated_data = [data_point + (span,) for data_point, span in zip(appended_pairs, halu_text_spans)]
    hard_spans = [get_overlap_span(data[1], data[5]) for data in updated_data]
    finished_data = [data_point + (span,) for data_point, span in zip(updated_data, hard_spans)]

    # Reformat and save output
    hard_labels = reformat(finished_data)
    with open("hard_labels.json", "w") as f:
        f.write('\n'.join(json.dumps(record) for record in hard_labels))

    # Evaluate performance
    ref_dict = load_jsonl_file_to_records(set_file)
    pred_dict = load_jsonl_file_to_records("hard_labels.json")
    ious = scorer(ref_dict, pred_dict)
    print(f"Average IoU: {np.mean(ious)}")
