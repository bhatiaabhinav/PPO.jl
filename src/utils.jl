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

function splitequal(H, TH)
    splits = []
    t = 0
    while t < H
        push!(splits, (t+1):min((t+TH), H))
        t += TH
    end
    return splits
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



get_rnn_state(m::Flux.Recur) = m.state
get_rnn_state(m) = map(get_rnn_state, Flux.functor(m)[1])


function set_rnn_state!(m::Flux.Recur, s::AbstractMatrix)
    if isa(m.state, Flux.CUDA.CuArray) && !isa(s, Flux.CUDA.CuArray)
        s = gpu(s)
    elseif !isa(m.state, Flux.CUDA.CuArray) && isa(s, Flux.CUDA.CuArray)
        s = cpu(s)
    end
    m.state = s
    nothing
end
function set_rnn_state!(m::Flux.Recur, s::AbstractMatrix, idx)
    if !isnothing(idx)
        if size(s, 2) != 1
            s = s[:, idx]
        end
    end
    set_rnn_state!(m, s)
end

function set_rnn_state!(m, state, idx=nothing)
    foreach(zip(Flux.functor(m)[1], state)) do (_m, _state)
        set_rnn_state!(_m, _state, idx)
    end
    nothing
end


function reset_rnn_state!(rnn)
    Flux.reset!(rnn)
    nothing
end

"""reset rnn state for just idx-th index in the batch. idx can also be a bool array marking all the indices for which the rnn state needs to be reset."""
function reset_rnn_state!(m::Flux.Recur, idx::BitVector)
    if all(idx)
        Flux.reset!(m)
        return nothing
    end
    if !any(idx)
        return nothing
    end
    n = size(m.state)[end]  # batch size
    mask = Flux.Zygote.ignore() do
        mask = zeros(eltype(m.state), 1, n)
        mask[1, idx] .= 1
        if m.state isa Flux.CUDA.CuArray
            mask = gpu(mask)
        end
        return mask
    end
    m.state = m.state .* (1 .- mask) + m.cell.state0 * mask
    nothing
end
function reset_rnn_state!(m::Flux.Recur, idx::Int)
    n = size(m.state)[end]  # batch size
    mask = Flux.Zygote.ignore() do
        mask = zeros(eltype(m.state), 1, n)
        mask[1, idx] = 1
        if m.state isa Flux.CUDA.CuArray
            mask = gpu(mask)
        end
        return mask
    end
    m.state = m.state .* (1 .- mask) + m.cell.state0 * mask
    nothing
end

function reset_rnn_state!(m, idx::BitVector)
    if all(idx)
        Flux.reset!(m)
        return nothing
    end
    if !any(idx)
        return nothing
    end
    foreach(_m -> reset_rnn_state!(_m, idx), Flux.functor(m)[1])
    nothing
end
function reset_rnn_state!(m, idx::Int)
    foreach(_m -> reset_rnn_state!(_m, idx), Flux.functor(m)[1])
    nothing
end


function decide_context_length(sl, FC, CL, TH)
    _CL = CL
    if FC > CL
        if sl >= FC - CL/2
            _CL = sl - (FC-CL)
        else
            _CL = CL ÷ 2
        end
    end
    if _CL % TH != 0
        _CL = _CL + TH - _CL % TH # so that _CL is divisible by TH
    end
    return _CL
end
