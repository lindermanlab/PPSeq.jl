# === Misc Utility Functions === #
const LOG_2_PI = log(2 * pi)

"""Log-gamma function."""
lgamma(x) = logabsgamma(x)[1]


"""
Sample from a list of log probabilities. Overwrites
vector so that no new memory is allocated.
"""
sample_logprobs!(log_probs) = sample(pweights(softmax!(log_probs)))


"""
Log normalizer of 1D normal distribution in information form.
given potential `m` and precision `v`. 

N(x | m, v) = exp{ m⋅x - .5v⋅x² - log Z(m, v) }
"""
gauss_info_logZ(m, v) = 0.5 * (LOG_2_PI - log(v) + m * m / v)
