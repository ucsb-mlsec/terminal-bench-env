Beta-VAE Disentanglement Experiment Overview

This experiment investigated the effects of different hyperparameter configurations on the 
disentanglement performance of beta-VAE models. The goal was to identify which combination 
of parameters produces the best disentangled representations.

Dataset:
A controlled shapes dataset was used, featuring 4 independent factors of variation:
- Shape (circle, square, triangle)
- Scale (size of the object)
- Orientation (rotation angle)
- Position (x, y coordinates)

Experimental Setup:
Multiple training runs were conducted, each testing different combinations of three key 
hyperparameters:
1. beta: The weight applied to the KL divergence term in the loss function (controls 
   disentanglement pressure)
2. learning_rate: The optimization learning rate
3. latent_dim: The dimensionality of the latent representation space

All training runs have been completed and disentanglement quality was evaluated using 
standard metrics (MIG, SAP, DCI scores). The results are deterministic and recorded.

Directory Structure:
- experiments/ contains subdirectories for each run (run_001/, run_002/, etc.)
- Each run directory includes:
  * config.json: Hyperparameters used for that specific run
  * metrics.log: Training and validation metrics over time
  * final_scores.txt: Final disentanglement evaluation results

Task:
Analyze the training artifacts to identify which configuration achieved the highest 
disentanglement score.