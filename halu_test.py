from transformers import pipeline, AutoTokenizer
import json

# Read data 
pairs = []
with open('test_data.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        pairs.append((data['model_output_text'], data['target_text']))

# Prompt the pairs
prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
input_pairs = [prompt.format(text1=pair[0], text2=pair[1]) for pair in pairs]

# Use text-classification pipeline to predict
classifier = pipeline(
    "text-classification",
    model='vectara/hallucination_evaluation_model',
    tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
    trust_remote_code=True
)
full_scores = classifier(input_pairs, top_k=None)  # List[List[Dict[str, float]]]

# Optional: Extract the scores for the 'consistent' label
simple_scores = [
    score_dict['score']
    for score_for_both_labels in full_scores
    for score_dict in score_for_both_labels
    if score_dict['label'] == 'consistent'
]

print(simple_scores)
