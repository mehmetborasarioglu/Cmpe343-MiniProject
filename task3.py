import numpy as np
from scipy.stats import t

#define the mean and the covariance matrix that was given in the project description
mu = np.array([20, 0.3, 0.8])
cov = np.array([[4, 0.5, 0.2], [0.5, 0.7, 0.2], [0.2, 0.2, 0.1]])

#function that represents the number of failures depending on the average temperature, the average server load, and the cooling efficiency
def g(x):
    return 0.1 * x[0]**2 + 12.5 * x[1]**2 - 7.5 * x[2]**2

#function for monte carlo sampling
def monte_carlo_sampling(n_samples, mu, cov):
    #we used the numpy functions for convinience
    samples = np.random.multivariate_normal(mu, cov, n_samples)
    
    g_values = []
    
    #for every sample generated calculate the failure count
    for sample in samples:
        g_value = g(sample) 
        g_values.append(g_value)
    
    #lets convert g_values into a numpy array to use numpy's functions
    g_values = np.array(g_values)
    
    #mean of failures
    mean_g = np.mean(g_values)
    
    #lets calculate the standart deviation of failures with 1 less degree of freedom
    std_g = np.std(g_values, ddof=1)  
    
    return mean_g, std_g

#we have predefined z as 1.96 since we will investigate the 95% confidence interval
def confidence_interval(mean, std, n_samples, z=1.96):
    #maximum possible error for this confidence interval and sample
    margin_of_error = z * (std / np.sqrt(n_samples))
    
    #our confidence interval
    return mean - margin_of_error, mean + margin_of_error

def hypothesis_test_single_sample(mu0, g1, s1, n1, alpha=0.05):
    #lets calculate the t
    t_stat = (g1 - mu0) / (s1 / np.sqrt(n1))
    
    #calculate the degree of freedom from the formula in slides
    df = n1 - 1
    
    #calculate t
    t_crit = t.ppf(1 - alpha / 2, df)
    
    # Decision rule
    if abs(t_stat) > t_crit:
        decision = "Reject H0"
    else:
        decision = "Fail to Reject H0"
    
    return t_stat, t_crit, df, decision

#our sample sizes
n_values = [50, 100, 1000, 10000]

#monte carlo sampling for each sample size
results = []
for n in n_values:
    mean, std = monte_carlo_sampling(n, mu, cov)
    
    #our confidance interval
    ci = confidence_interval(mean, std, n)
    results.append((n, mean, std, ci))

#comparison between sample size 50 and sample size 10 000
n0, n1 = 10000, 50
g0_mean, g0_std = monte_carlo_sampling(n0, mu, cov)
g1_mean, g1_std = monte_carlo_sampling(n1, mu, cov)

# Perform single-sample t-test
t_stat, t_crit, df, decision = hypothesis_test_single_sample(g0_mean, g1_mean, g1_std, n1)

print("Monte Carlo Sampling Results:")
for n, mean, std, ci in results:
    print(f"Sample Size: {n}")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std Dev: {std:.4f}")
    print(f"  95% Confidence Interval: {ci}")
    print()

print("\nHypothesis Testing Results (Single-Sample t-Test):")
print(f"Population Mean (mu0): {g0_mean:.4f}")
print(f"Sample Mean (g1): {g1_mean:.4f}, Sample Std Dev (s1): {g1_std:.4f}")
print(f"t-statistic: {t_stat:.4f}, critical t-value: {t_crit:.4f}, degrees of freedom: {df}")
print(f"Decision: {decision}")
