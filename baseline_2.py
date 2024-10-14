from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import json

# Load the fine-tuned BERT model and tokenizer for span-based hallucination detection
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

    # Tokenize the input text for BERT with offset mapping
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, return_offsets_mapping=True)
    
    offset_mapping = inputs['offset_mapping'][0]  # Get offset mappings for tokens

    # Remove offset mapping from inputs as the model doesn't need it
    inputs.pop('offset_mapping')

    # Perform inference
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the most likely start and end positions for the hallucinated span
    start_idx = torch.argmax(start_scores).item()
    end_idx = torch.argmax(end_scores).item()

    # Get token indices and map them back to character indices using offset mapping
    if start_idx <= end_idx:
        start_char = offset_mapping[start_idx][0]
        end_char = offset_mapping[end_idx][1]

        # Extract the predicted span text from the hypothesis
        predicted_span_text = pair[2][start_char:end_char]
    else:
        predicted_span_text = None
        start_char, end_char = -1, -1  # If invalid span, mark it

    # Print useful information for debugging
    print(f"Input Text: {input_text}")
    print(f"Predicted Span Text: '{predicted_span_text}'")
    print(f"Start Char Index: {start_char}, End Char Index: {end_char}")

    if start_char != -1 and end_char != -1 and start_char < end_char:
        predicted_spans.append({
            'id': pair[0],  # Include the id from input data
            'model_output_text': pair[1],
            'target_text': pair[2],
            'predicted_span': predicted_span_text,
            'hard_labels': [{'start': start_char, 'end': end_char}],
            'soft_labels': pair[3]
        })
    else:
        # Handle case where no valid span was found
        predicted_spans.append({
            'id': pair[0],
            'model_output_text': pair[1],
            'target_text': pair[2],
            'predicted_span': None,
            'hard_labels': [],
            'soft_labels': pair[3]
        })

# Print or save the predicted spans with indices
for prediction in predicted_spans:
    print(json.dumps(prediction, indent=2))

# Save the results to a JSONL file
with open('predictions.jsonl', 'w') as outfile:
    for prediction in predicted_spans:
        json.dump(prediction, outfile)
        outfile.write("\n")
