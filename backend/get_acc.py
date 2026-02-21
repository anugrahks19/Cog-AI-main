import sys
import evaluate_model

orig_stdout = sys.stdout
with open("clean_eval.txt", "w", encoding="utf-8") as f:
    sys.stdout = f
    evaluate_model.evaluate()
sys.stdout = orig_stdout
