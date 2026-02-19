# Load required library
library(lme4)

# Read the data
data <- read.csv('data/experiment_data.csv')

# Scale reaction time to improve numerical stability
data$reaction_time_scaled <- scale(data$reaction_time)

# Fit the mixed-effects logistic regression model
model <- glmer(accuracy ~ reaction_time_scaled + (1|participant_id) + (1|site_id), 
               data=data, 
               family=binomial)

# Print model summary
summary(model)