#Installation

1. run -> python3 baseline.py
2. New jsonl file is created called predictions.jsonl
3. Use this file together with the sample_test.v1.json file to run the scorer.py to create the evaluation (called test_scores) -> python3 scorer.py sample_test.v1.json predictions.jsonl test_scores  
