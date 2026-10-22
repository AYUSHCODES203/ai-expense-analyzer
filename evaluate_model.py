
from ml_model import evaluate_model

accuracy, report = evaluate_model()

print("=" * 55)
print("EXPENSE CATEGORIZATION MODEL EVALUATION")
print("=" * 55)
print(f"\nTest accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(report)
