
"""Sort neurons by preferred sequence type and offset."""
function sortperm_neurons(globals::SeqGlobals; thres=0.2)

    resp_props = exp.(globals.neuron_response_log_proportions)
    offsets = globals.neuron_response_offsets
    peak_resp = dropdims(maximum(resp_props, dims=2), dims=2)

    has_response = peak_resp .> quantile(peak_resp, thres) 
    preferred_type = [idx[2] for idx in argmax(resp_props, dims=2)]
    preferred_delay = [offsets[n, r] for (n, r) in enumerate(preferred_type)]

    Z = collect(zip(has_response, preferred_type, preferred_delay))
    return sortperm(Z)
end
