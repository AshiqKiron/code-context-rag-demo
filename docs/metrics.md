# Metrics Framework: AI Performance Evaluation

To ensure our AI system is improving, we track three core metrics using our Golden Dataset.

## 1. Accuracy
- **Definition**: Does the generated code actually solve the user's problem?
- **Measurement**: Binary check (Pass/Fail) against expected outcomes in the Golden Dataset.
- **Target**: > 90%

## 2. Context Adherence
- **Definition**: Did the model use the specific rules from our `AuthService` or `PaymentService` instead of making up generic code?
- **Measurement**: Keyword matching and semantic similarity between retrieved context and generated output.
- **Target**: > 95%

## 3. Syntax Error Rate
- **Definition**: How often does the model produce code that doesn't compile?
- **Measurement**: Automated linting of generated outputs.
- **Target**: < 2%

## Why These Metrics Matter
Generic models often have high "fluency" but low "factuality." By tracking **Context Adherence**, we ensure the model isn't just writing *good* Python, but writing *our* Python.
