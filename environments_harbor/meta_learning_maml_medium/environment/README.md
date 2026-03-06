# MAML Regression Task

Complete the evaluation code in maml_regression.py to test the meta-learned model.

The model has been meta-trained and is ready for evaluation.
You need to implement the evaluation logic that tests adaptation on new tasks.

Your evaluation should:
- Test on 5 new sinusoidal regression tasks
- Measure performance before and after adaptation
- Use 10 support examples for adaptation with 3 gradient steps
- Report average MSE losses in /workspace/results.txt