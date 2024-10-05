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


function cartesian_product(X::AbstractArray{Int}, Y::AbstractArray{Int})
    return [CartesianIndex(x, y) for x in X, y in Y]
end

function cartesian_product(Xdims::Int, Ydims::Int)
    return cartesian_product(1:Xdims, 1:Ydims)
end



function rollouts_parallel(envs, nsteps, actor, device, rng, progressmeter=false; reset_all=true)

    reset!(envs, reset_all; rng=rng)

    state_dim = size(state_space(envs), 1)
    isdiscrete = action_space(envs) isa IntegerSpace
    if isdiscrete
        nactions = length(action_space(envs))
    else
        action_dim = size(action_space(envs), 1)
    end
    M, N = nsteps, length(envs)

    if state_dim > 50
        𝐬 = zeros(Float32, state_dim, M, N) |> device
    else
        𝐬 = zeros(Float32, state_dim, M, N)
    end
    if isdiscrete
        𝐚 = zeros(Int, 1, M, N)
    else
        𝐚 = zeros(Float32, action_dim, M, N)
    end
    𝐫 = zeros(Float32, 1, M, N)
    𝐭 = zeros(Float32, 1, M, N)
    𝐝 = zeros(Float32, 1, M, N)

    progress = Progress(M; color=:white, desc="Collecting trajectories", enabled=progressmeter)

    Flux.reset!(actor)
    for t in 1:M
        reset!(envs, false; rng=rng)
        𝐬ₜ = state(envs) |> tof32
        if state_dim > 50
            𝐬[:, t, :] .= device(𝐬ₜ)
        else
            𝐬[:, t, :] .= 𝐬ₜ
        end
        if isdiscrete
            @assert actor isa PPOActorDiscrete
            if actor.recurtype ∈ (MARKOV, RECURRENT)
                𝛑ₜ, log𝛑ₜ = get_probs_logprobs(actor, device(𝐬ₜ)) |> cpu
            elseif actor.recurtype == TRANSFORMER
                s_t = 𝐬[:, 1:t, :]
                if state_dim > 50
                    s_gpu = s_t
                else
                    s_gpu = s_t |> device
                end
                𝛑ₜ, log𝛑ₜ = get_probs_logprobs(actor, s_gpu) |> cpu
                𝛑ₜ, log𝛑ₜ = 𝛑ₜ[:, t, :], log𝛑ₜ[:, t, :]
                s_gpu = nothing
                s_t = nothing
            end
            𝐚ₜ = reshape([sample(rng, 1:nactions, ProbabilityWeights(𝛑ₜ[:, i])) for i in 1:N], 1, N)
            𝐚[:, t, :] = 𝐚ₜ
        else
            @assert actor isa PPOActorContinuous
            if actor.recurtype ∈ (MARKOV, RECURRENT)
                𝐚ₜ, log𝛑ₜ, log𝛔ₜ = sample_action_logprobs(actor, rng, device(𝐬ₜ); return_logstd=true) |> cpu
            else
                𝐚ₜ, log𝛑ₜ, log𝛔ₜ = sample_action_logprobs(actor, rng, device(𝐬[:, 1:t, :]); return_logstd=true) |> cpu
                𝐚ₜ, log𝛑ₜ = 𝐚ₜ[:, end, :], log𝛑ₜ[:, end, :]
            end
            𝐚[:, t, :] = 𝐚ₜ
        end

        if isdiscrete
            _𝐚ₜ = 𝐚ₜ[1, :]
        else
            Tₐ = envs |> action_space |> eltype |> eltype
            _𝐚ₜ = convert(Matrix{Tₐ}, 𝐚ₜ) |> eachcol .|> copy    # eachcol makes it a vector of vectors
            _𝐚ₜ = convert(Matrix{Tₐ}, 𝐚ₜ)
        end
        step!(envs, _𝐚ₜ; rng=rng)
        𝐫ₜ = reward(envs)' |> tof32
        𝐫[:, t, :] = 𝐫ₜ
        𝐭ₜ = in_absorbing_state(envs)' |> tof32
        𝐭[:, t, :] = 𝐭ₜ
        𝐝ₜ = (in_absorbing_state(envs) .|| truncated(envs))' |> tof32
        𝐝[:, t, :] = 𝐝ₜ
        next!(progress)
    end
    finish!(progress)
    return 𝐬, 𝐚, 𝐫, 𝐭, 𝐝
end