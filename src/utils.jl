using Statistics
using Flux
using Flux: unsqueeze

function tof32(𝐱::AbstractArray{<:Real, N})::AbstractArray{Float32, N} where N
    convert(AbstractArray{Float32, N}, 𝐱)
end


const SQRT_2PI = Float32(sqrt(2π))
const LOG_SQRT_2PI = Float32(log(SQRT_2PI))
const LOG_2PI = Float32(log(2π))

@inline function log_nomal_prob(x::T, μ::T, σ::T)::T where {T<:Real}
    return -0.5 * ((x - μ) / σ)^2 - log(SQRT_2PI * σ)
end

@inline function log_nomal_prob(x::T, μ::T, σ::T, logσ::T)::T where {T<:Real}
    return -0.5 * ((x - μ) / σ)^2 - LOG_SQRT_2PI - logσ
end


batch(x) = reshape(x, size(x)..., 1)
batch(x::Real) = [x]
unbatch(x) = selectdim(x, ndims(x), 1)
unbatch(x::AbstractVector) = x[1]
unbatch_last(x) = selectdim(x, ndims(x), size(x, ndims(x)))
unbatch_last(x::AbstractVector) = x[end]


using Flux


"""
    clip gradients by global norm. Norm is calulated by concatenating all gradients into a single vector and then calculating the norm of that vector.

# Arguments
    gs: gradients dict. Access gradiens of parameter `p` by `gs[p]`
    ps: parameters list
    maxnorm: maximum norm allowed. If norm is greater than this, then all gradients are scaled down by the same factor so that the norm becomes equal to this value.
"""
function clip_global_norm!(gs, ps, maxnorm)
    norm = 0
    for p in ps
        !haskey(gs, p) && continue
        isnothing(gs[p]) && continue
        norm += sum(abs2, gs[p])
    end
    norm = sqrt(norm)
    if norm > maxnorm
        scale = maxnorm / norm
        for p in ps
            !haskey(gs, p) && continue
            isnothing(gs[p]) && continue
            gs[p] *= scale
        end
    end
    nothing
end


function make_adam_optim(lr, betas, epsilon, weight_decay)
    optim_chain = []
    if weight_decay > 0
        push!(optim_chain, WeightDecay(weight_decay))
    end
    push!(optim_chain, Adam(lr, betas, epsilon))
    return Flux.Optimiser(optim_chain...)
end
