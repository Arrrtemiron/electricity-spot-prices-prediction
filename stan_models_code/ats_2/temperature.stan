data {
  int<lower=0> N;  // Number of time points
  vector[N] y;     // Log prices
  vector[N] temp;   // Temperature in Fahrenheit
  vector[N] D_Sat;  // Indicator for Saturday
  vector[N] D_Sun;  // Indicator for Sunday
  vector[N] D_Mon;  // Indicator for Monday
}
parameters {
  real mu;
  real<lower=0, upper=2> kappa_h;
  real theta_h;
  real psi;
  real d_sat;
  real d_sun;
  real d_mon;
  real<lower=0, upper=1> rho_raw;
  vector[N] eps1;
  vector[N] eps2;
  simplex[3] p; // Dirichlet distribution for jumps
  real n_U;
  real n_D;
  vector<lower=0.1>[N] xi_d;
  vector<lower=0.1>[N] xi_u;
}
transformed parameters {
  vector[N] h; //parameter of model
  vector[N] j; //parameter of model

  real rho = 2 * rho_raw - 1; // (-1, 1)
  real sigma_h = 0.03;  //variance of log(y_t - y_{t-1}) of input data

  h[1] = 0;
  for (i in 2:N) {
    h[i] = h[i-1] + kappa_h * (theta_h - h[i-1]) + sigma_h * (rho * eps1[i] + sqrt(1 - rho^2) * eps2[i]);
  }
}
model {
  // Y VARS
  mu ~ normal(0, 0.8);
  d_sat ~ normal(0, 0.8);
  d_sun ~ normal(0, 0.8);
  d_mon ~ normal(0, 0.8);
  psi ~ normal(0, 0.8);
  // H VARS
  kappa_h ~ normal(1, 4.5);
  theta_h ~ normal(0, 0.8);
  eps1 ~ normal(0, 1);
  eps2 ~ normal(0, 1);
  rho_raw ~ beta(5, 5);
  // JUMP VARS
  p ~ dirichlet([1,1,1]');
  n_U ~ inv_gamma(1.86, 0.43);
  n_D ~ inv_gamma(1.86, 0.43);
  xi_d ~ exponential(n_D);
  xi_u ~ exponential(n_U);
}
generated quantities {
  vector[N] y_pred_train;
  vector[N] jumps;
  vector[N] q;
  real prediction;
  {
    for (i in 1:N) {
      q[i] = categorical_rng(p) - 2;
    }
  }
  {
    for (i in 1:N) {
      if (q[i] == -1)
        jumps[i] = -1 * xi_d[i];
      else if (q[i] == 0)
        jumps[i] = 0;
      else
        jumps[i] = xi_u[i];
    }
  }
  {
    y_pred_train[1] = y[1];
    for (i in 2:N-1) {
        y_pred_train[i] = y[i-1] + mu + d_sat * D_Sat[i] + d_sun * D_Sun[i] + d_mon * D_Mon[i] + sqrt(exp(h[i-1])) * eps1[i] + jumps[i] + temp[i] * psi;
    }
  }
  prediction = y_pred_train[N-1] + mu + d_sat * D_Sat[N] + d_sun * D_Sun[N] + d_mon * D_Mon[N] + sqrt(exp(h[N-1])) * eps1[N] + jumps[N] + temp[N] * psi;
}