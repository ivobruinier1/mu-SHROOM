# Installation

1. Install requirements -> pip install -r requirements.txt
2. run -> python3 baseline.py
3. New jsonl file is created called predictions.jsonl
4. Use this file together with the sample_test.v1.json file to run the scorer.py to create the evaluation (called test_scores) -> python3 scorer.py sample_test.v1.json predictions.jsonl test_scores  
