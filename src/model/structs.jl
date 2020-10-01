"""
Holds neuron index and timestamp for each spike.
"""
struct Spike
    neuron::Int64
    timestamp::Float64
end

"""
Summary information for latent events. See `SeqEvent` for
the full-featured struct that caches things like sufficient
statistics. This summary holds the basic information that
is saved during MCMC sampling and output to the user.

assignment_id :

    Integer specifying the cluster id. Corresponds to
    the `indices` in `SeqEventList` struct.

timestamp :

    Time that the latent event occured. See
    `sampled_timestamp` in `SeqEvent`.

seq_type :

    Integer in {1, 2, ..., R} specifying the sequence
    type. See `sampled_type` in `SeqEvent` struct.

seq_warp:
    
    Float in [1/τ, ..., τ] specifying the inferred warp
    of the sequence.  See `sampled_warp` in `SeqEvent`;
    here we report the warp value rather than its index.

amplitude :

    Sampled amplitude of the latent event (i.e. the
    expected number of evoked spikes). See
    `sampled_amplitude` in `SeqEvent` struct.
"""
struct EventSummaryInfo
    assignment_id::Int64
    timestamp::Float64
    seq_type::Int64
    seq_warp::Float64
    amplitude::Float64
end


"""
Prior distributions for the SeqModel model.

seq_event_rate :

    Rate parameter, λ, for latent events. The number of
    sequence events, K, follows:

        K ~ Poisson(λ * max_time)

seq_type_proportions :
    Dirichlet prior over the relative proportions
    of R sequence event types.

        π ~ Dirichlet(γ * 1_R)

seq_event_amplitude :

    RateGamma prior on the amplitude (i.e., expected number of
    evoked spikes) of each sequence event.

        A_k ~ RateGamma(α, β)

neuron_response_proportions :

    Dirichlet prior over the assignment probabilities of
    sequence-evoked spikes across neurons.

        a ~ Dirichlet(φ * 1_N)

neuron_response_profile :

    Normal-Inverse-Chi-Squared distribution over the
    mean and variance (i.e. the delay and width) of
    each neuron's respons to a latent event.

        c_nr ~ Inv-χ²(ν, s)
        b_nr ~ Normal(0, c_nr / κ)

bkgd_amplitude :
    RateGamma prior on the rate parameter 

        A_∅ ~ RateGamma(α_∅, β_∅)
    
    The number of background then follows a homogeneous
    Poisson process over the interval [0, T] with
    intensity A_bkgd.

bkgd_proportions :

    Dirichlet prior over the assignment probabilities of
    background spikes to neurons.

        a_∅ ~ Dirichlet(φ_∅ * 1_N)

warp_values :
    
    Length W vector of warp values, typically log-spaced
    from 1/τ to τ for some τ > 1.

warp_log_proportions :
    
    Length W log-probability vector over warp values.
"""
struct SeqPriors
    seq_event_rate::Float64

    seq_type_proportions::SymmetricDirichlet
    seq_event_amplitude::RateGamma
    
    neuron_response_proportions::SymmetricDirichlet
    neuron_response_profile::NormalInvChisq

    bkgd_amplitude::RateGamma
    bkgd_proportions::SymmetricDirichlet

    warp_values::Vector{Float64}
    warp_log_proportions::Vector{Float64}
end


"""
PP-Seq Global Parameters.

seq_type_log_proportions :
    Log-probability vector over sequence types. Element r
    is the probability that a sequence event has type r.

neuron_response_log_proportions :
    N × R matrix, each column is a log-probability vector over
    neurons. Element (n, r) is proportional to the response
    amplitude of neuron n to sequences of type r. Column r is
    denoted `log (a_r)` in the paper.

neuron_response_offsets :
    N × R matrix. Element (n, r) is the mean of the
    univariate Gaussian response profile for neuron n
    to sequences of type r.

neuron_response_widths :
    N × R matrix. Element (n, r) is the variance of the
    univariate Gaussian response profile for neuron n
    to sequences of type r.

bkgd_amplitude :
    The expected number of spikes (pooled across all neurons)
    in a unit time interval. Denoted `A_∅`
    in the paper.

bkgd_neuron_log_proportions :
    Log-probability vector over neurons. Element n is the
    log-probability that a background spike comes from neuron
    n, when chosen randomly. Denoted `a_∅` in the paper.
"""
mutable struct SeqGlobals
    seq_type_log_proportions::Vector{Float64} # R

    neuron_response_log_proportions::Matrix{Float64}  # N × R
    neuron_response_offsets::Matrix{Float64}  # N × R
    neuron_response_widths::Matrix{Float64}  # N × R

    bkgd_amplitude::Float64
    bkgd_log_proportions::Vector{Float64} # N

end


"""
Holds sufficient statistics needed to estimate the time,
amplitude, and type of a sequence occurence / latent event.

Parameters
----------
spike_count :
    Number of spikes assigned to this sequence.

summed_potentials :
   g Size RxW matrix, element (r,w) stores,

        Σᵢ mᵢ

        where mᵢ = (tᵢ - b_{nᵢ,r} * τ_w) / (c_{nᵢ,r} * τ_w)

    where the sum is over spikes (tᵢ, nᵢ) assigned to
    this sequence event. The parameters b_{r,nᵢ} and
    c_{r,nᵢ} correspond to the response offset and width
    of neuron nᵢ to sequences of type r, and they are each
    linearly scaled by the warp value τ_w.

summed_precisions :
    Size RxW matrix, element (r,w) stores,

        Σᵢ vᵢ

        where vᵢ = 1 / (c_{nᵢ,r} * τ_w)

    where the sum is over spikes (tᵢ, nᵢ) assigned to
    this sequence event. The parameter c_{r,nᵢ}
    corresponds to the response width of neuron nᵢ
    to sequences of type r, and it is linearly scaled
    by the warp value τ_w.

summed_logZ:
    Size RxW matrix, element (r,w) stores,

        Σᵢ log[ a_{nᵢ,r} / Z(mᵢ, vᵢ) ]

    where the sum is over spikes (tᵢ, nᵢ) assigned to
    this sequence event. The parameter a_{nᵢ,r} is
    the response amplitude of neuron nᵢ to sequences
    of type r. The term `Z(mᵢ, vᵢ)` denotes the log
    normalization constant for a univariate Gaussian
    with precision `vᵢ` and potential `mᵢ`.  Note that
    the precision and linear potentials depend on τ_w.

    TODO: This naming convention is a little strange
    since we're really storing log a -log Z.

seq_type_posterior:
    Size RxW matrix, element (r,w) is proportional to:

        Σᵢ(log a_{nᵢ,r} - Σᵢ log Z(mᵢ, vᵢ)) + log Z(Σᵢ mᵢ, Σᵢ vᵢ)

    After normalization, the represents the posterior distribution
    over the event sequence type and warp value, given the spikes 
    currently assigned to this event.

sampled_type :
    The current type of the event (changes over
    Gibbs samples).

sampled_warp:
    The current warp index of the event (changes over
    Gibbs samples).

sampled_timestamp :
    The current time of the event (changes over
    Gibbs samples).

sampled_amplitude :
    The current amplitude of the event (changes over
    Gibbs samples).
"""
mutable struct SeqEvent
    spike_count::Int64
    summed_potentials::Matrix{Float64} # RxW
    summed_precisions::Matrix{Float64} # RxW
    summed_logZ::Matrix{Float64} # RxW
    seq_type_posterior::Matrix{Float64} # RxW

    sampled_type::Int64
    sampled_warp::Int64
    sampled_timestamp::Float64
    sampled_amplitude::Float64
end


"""
Dynamically re-sized array holding SeqEvent structs.

events :
    Vector of SeqEvent structs, some may be empty.

indices :
    Sorted vector of integers, specifying the indices of
    non-empty SeqEvent structs. Does not contain duplicate
    integers. Note that `length(indices) <= length(events)`,
    with equality only holding if there are no empty events.
"""
struct SeqEventList
    events::Vector{SeqEvent}
    indices::Vector{Int64}    # Indices of occupied sequences.
end


"""
Point Process Sequence Detection (PP-Seq) Model.

max_time :
    All spikes and sequences occur on the time interval
    between zero and max_time.

max_sequence_length :
    Maximum length of a sequence (used to speed up
    parent assignment step of collapsed Gibbs sampling
    -- we don't compute statistics for sequences larger
    than this threshold away from the spike.)

priors :
    Prior distributions.

globals :
    Global variables.

sequence_events :
    List of SeqEvent structs. See `./events.jl` for functionality.

_R_buffer :
    Vector of size `num_sequence_types`.

_RW_buffer :
    Matrix of size `num_sequence_types x num_warps`.

_K_buffer : 
    Resized vector, holding probabilities over the number of latent events. 
"""
mutable struct SeqModel
    max_time::Float64
    max_sequence_length::Float64

    priors::SeqPriors
    globals::SeqGlobals
    sequence_events::SeqEventList

    # TODO : move these to globals
    new_cluster_log_prob::Float64
    bkgd_log_prob::Float64

    _R_buffer::Vector{Float64}
    _RW_buffer::Matrix{Float64}
    _K_buffer::Vector{Float64}
end


"""
Distributed Point Process Sequence Model
"""
mutable struct DistributedSeqModel
    primary_model::SeqModel
    num_partitions::Int64
    submodels::Vector{SeqModel}    
end
