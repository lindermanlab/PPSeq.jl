module PPSeq

# === IMPORTS === #

using LinearAlgebra
using Statistics

import Base: size, rand, length, getindex, iterate
import Base.Iterators
import StatsBase: sample, pweights, denserank, mean
import Random
import SpecialFunctions: logabsgamma, logfactorial
import StatsFuns: softmax!, softmax, logaddexp, logsumexp, normlogpdf, normpdf
import Distributions
import Distributions: cdf
import PyPlot: plt

# === EXPORTS === #

export Spike, EventSummaryInfo, SeqHypers, SeqGlobals, SeqModel, DistributedSeqModel
export log_joint, log_prior, log_p_latents
export SymmetricDirichlet, RateGamma, ScaledInvChisq, NormalInvChisq
export specify_gamma, sample
export split_merge_sample!, gibbs_sample!, annealed_gibbs!
export masked_gibbs!, annealed_masked_gibbs!, Mask
export create_random_mask, split_spikes_by_mask, create_blocked_mask, sample_masked_spikes!

# === UTILS === #

include("./utils/misc.jl")
include("./utils/distributions.jl")

# === MODEL === #

 # Model structs
 # - SeqModel : holds full model
 # - SeqEvent : holds suff stats for a latent event
 # - SeqEventList : dynamically re-sized array of latent events.
 # - SeqPriors : holds prior distributions.
 # - SeqGlobals : holds global variables.
include("./model/structs.jl")

 # Convience methods for constructing / accessing SeqModel model struct.
include("./model/model.jl")

 # Distributed version of the model
include("./model/distributed.jl")

 # Methods for creating latent events, managing their storage.
include("./model/events.jl")

 # Adds and removes spikes from events, updates to sufficient statistics.
include("./model/add_remove.jl")

 # Evaluate various probability distributions for the model:
 # - predictive posterior
 # - prior distribution on global parameters
 # - distribution on latent events, given global parameters
 # - joint distribution on global parameters, latent events, observed spikes
include("./model/probabilities.jl")

 # Evaluate log likelihood on heldout data.
include("./model/masked_probabilities.jl")

# === PARAMETER INFERENCE === #

 # Collapsed Gibbs sampling.
include("./algorithms/gibbs.jl")

 # Masked Gibbs sampling (for cross-validation).
include("./algorithms/masked_gibbs.jl")

 # Distributed collapsed Gibbs sampling
include("./algorithms/distributed_gibbs.jl")

 # Collapsed Gibbs sampling with annealing.
include("./algorithms/annealed_gibbs.jl")

 # Split merge sampler.
include("./algorithms/split_merge.jl")

 # Easy one-stop-shop sampling function.
include("./algorithms/easy_sample.jl")

# === USER-FACING UTILS AND HELPER FUNCTIONS === #

 # Model configuration.
include("./utils/config.jl")

 # PyPlot visualizations.
include("./utils/visualization.jl")

 # Data analysis functions.
include("./utils/analysis.jl")

# === END OF MODULE === #

end
