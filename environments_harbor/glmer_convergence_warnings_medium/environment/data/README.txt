MULTI-SITE COGNITIVE EXPERIMENT DATA DOCUMENTATION
=====================================================

Overview:
---------
This dataset contains reaction time and accuracy data from a multi-site cognitive 
experiment. The data is structured for hierarchical analysis using mixed-effects 
models to account for the nested structure of observations.

Data Structure:
---------------
The experiment_data.csv file contains 5 columns:

1. participant_id
   - Unique identifier for each participant
   - Format: Alphanumeric codes
   - Multiple observations per participant (repeated measures)

2. site_id
   - Identifier for the testing site where data was collected
   - Format: Alphanumeric codes
   - Multiple participants are nested within each site

3. trial_number
   - Sequential trial number for each participant
   - Represents the order of stimulus presentation
   - Each participant completed multiple trials

4. reaction_time
   - Response latency measured in milliseconds (ms)
   - Range: 200-2000 ms
   - Continuous predictor variable

5. accuracy
   - Binary outcome indicating response correctness
   - Values: 1 = correct response, 0 = incorrect response
   - Dependent variable for the analysis

Hierarchical Structure:
-----------------------
- Level 1: Trials (multiple trials per participant)
- Level 2: Participants (multiple participants per site)
- Level 3: Sites (multiple testing locations)

Intended Analysis:
------------------
Mixed-effects logistic regression examining how reaction time predicts accuracy
while accounting for random variation across participants and sites.