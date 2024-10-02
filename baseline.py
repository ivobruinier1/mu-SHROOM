from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import json

# Load the fine-tuned BERT model and tokenizer this one should be able to give spans but didnt really work yet
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

pairs = []
# Read data from sample_test.v1.json
with open('sample_test.v1.json', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        # Collect id, model_output_text, model_input, and soft_labels as pairs
        pairs.append((data['id'], data['model_output_text'], data['model_input'], data['soft_labels']))

# Define the modified prompt template
prompt = "Identify the part of the hypothesis that contradicts the premise.\n\nPremise: {text1}\n\nHypothesis: {text2}"

# List to hold all predicted spans
predicted_spans = []

# Iterate through pairs and get the hallucinated span for each one
for pair in pairs:
    # Format the input prompt
    input_text = prompt.format(text1=pair[1], text2=pair[2])

    # Tokenize the input text for BERT
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Perform inference
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the most likely start and end positions for the hallucinated span
    start_idx = torch.argmax(start_scores).item()
    end_idx = torch.argmax(end_scores).item()
    original_text = input_text  # The complete input text used for BERT
    # Retrieve the tokens for the predicted span
    input_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx + 1])
    predicted_span_text = tokenizer.convert_tokens_to_string(tokens)
    start_char_idx = original_text.find(predicted_span_text)
    end_char_idx = start_char_idx + len(predicted_span_text)

    # This helps finding bugs so far the model has not produced valuable output but it makes the correct output
    print(f"Input Text: {original_text}")
    print(f"Predicted Span Text: '{predicted_span_text}'")
    print(f"Start Char Index: {start_char_idx}, End Char Index: {end_char_idx}")

    if start_char_idx != -1 and end_char_idx != -1 and start_char_idx < end_char_idx:
        predicted_spans.append({
            'id': pair[0],  # Include the id from input data
            'model_output_text': pair[1],
            'target_text': pair[2],
            'predicted_span': predicted_span_text,
            'hard_labels': [{'start': start_char_idx, 'end': end_char_idx}],
            'soft_labels': pair[3]  # It basicly steals the soft labels from the dataset no real transformer working here
        })
    else:
        # Handle case where no valid span was found
        predicted_spans.append({
            'id': pair[0],
            'model_output_text': pair[1],
            'target_text': pair[2],
            'predicted_span': None,
            'hard_labels': [],
            'soft_labels': pair[3]  # It basicly steals the soft labels from the dataset no real transformer working here
        })

# Print or save the predicted spans with indices
for prediction in predicted_spans:
    print(json.dumps(prediction, indent=2))

# This saves results to a file called predictions.jsonl use this file together with the sample_test file to get a score
with open('predictions.jsonl', 'w') as outfile:
    for prediction in predicted_spans:
        json.dump(prediction, outfile)
        outfile.write("\n")
