from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch
import json
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the fine-tuned BERT model and tokenizer for span-based hallucination detection
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)  # Use BertTokenizerFast

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

# Helper functions for evaluation
def compute_exact_match(pred_start, pred_end, true_start, true_end):
    """Check if predicted span exactly matches the ground truth span."""
    return int(pred_start == true_start and pred_end == true_end)

def compute_f1(pred_start, pred_end, true_start, true_end):
    """Compute the F1 score for a predicted span vs ground truth span."""
    pred_span = set(range(pred_start, pred_end + 1))
    true_span = set(range(true_start, true_end + 1))
    overlap = pred_span.intersection(true_span)
    
    if len(overlap) == 0:
        return 0.0
    precision = len(overlap) / len(pred_span)
    recall = len(overlap) / len(true_span)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

# Evaluation metrics storage
exact_matches = 0
total_examples = 0
f1_scores = []

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

        # Evaluation step
        true_start = pair[3][0]['start']
        true_end = pair[3][0]['end']

        # Compute exact match and F1
        exact_match = compute_exact_match(start_char, end_char, true_start, true_end)
        exact_matches += exact_match

        f1 = compute_f1(start_char, end_char, true_start, true_end)
        f1_scores.append(f1)

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

    total_examples += 1

# Print the final evaluation results
average_f1 = sum(f1_scores) / total_examples
exact_match_rate = exact_matches / total_examples

print(f"Exact Match Rate: {exact_match_rate * 100:.2f}%")
print(f"Average F1 Score: {average_f1 * 100:.2f}%")

# Save the results to a JSONL file
with open('predictions.jsonl', 'w') as outfile:
    for prediction in predicted_spans:
        json.dump(prediction, outfile)
        outfile.write("\n")
