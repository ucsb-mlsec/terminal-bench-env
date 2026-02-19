// Bayesian hierarchical model for grouped data
// This implementation uses centered parameterization which can cause
// numerical instability issues with the sampler

data {
  int<lower=1> N;                    // number of observations
  int<lower=1> J;                    // number of groups
  int<lower=1,upper=J> group[N];     // group indicator for each observation
  vector[N] y;                       // observed outcomes
}

parameters {
  vector[J] theta;                   // group-level parameters (centered)
  real mu;                           // population mean
  real<lower=0> tau;                 // between-group standard deviation
  real<lower=0> sigma;               // within-group standard deviation
}

model {
  // Priors
  mu ~ normal(0, 10);
  tau ~ cauchy(0, 5);
  sigma ~ cauchy(0, 5);
  
  // Hierarchical structure - centered parameterization
  // This causes numerical problems when tau is small relative to the data
  theta ~ normal(mu, tau);
  
  // Likelihood
  for (n in 1:N) {
    y[n] ~ normal(theta[group[n]], sigma);
  }
}

generated quantities {
  // Compute some posterior predictive quantities
  vector[N] y_rep;
  for (n in 1:N) {
    y_rep[n] = normal_rng(theta[group[n]], sigma);
  }
}