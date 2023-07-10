using MDPs
using StatsBase
using Flux
using Random

export PPOActor, PPOActorDiscrete, PPOActorContinuous, RecurrenceType, MARKOV, RECURRENT, TRANSFORMER

@enum RecurrenceType MARKOV RECURRENT TRANSFORMER


mutable struct PPOActorDiscrete{T<:AbstractFloat} <: AbstractPolicy{Vector{T}, Int}
    const recurtype::RecurrenceType
    actor_model  # maps states to action log probabilities
    deterministic::Bool
    const n::Int # number of actions

    const observation_history::Vector{Vector{Float32}}
    const misc_args
end
function PPOActorDiscrete{T}(recurtype::RecurrenceType, actor_model, deterministic::Bool, n::Int, misc_args) where {T}
    return PPOActorDiscrete{T}(recurtype, actor_model, deterministic, n, Vector{Float32}[], misc_args)
end

Flux.@functor PPOActorDiscrete (actor_model, )
Flux.gpu(p::PPOActorDiscrete{T}) where {T}  = PPOActorDiscrete{T}(p.recurtype, Flux.gpu(p.actor_model), p.deterministic, p.n, p.observation_history, p.misc_args)
Flux.cpu(p::PPOActorDiscrete{T}) where {T}  = PPOActorDiscrete{T}(p.recurtype, Flux.cpu(p.actor_model), p.deterministic, p.n, p.observation_history, p.misc_args)



function (p::PPOActorDiscrete{T})(rng::AbstractRNG, 𝐬::AbstractArray{Float32})::VecOrMat{Int} where {T}
    probabilities, logprobabilities = get_probs_logprobs(p, 𝐬)
    if ndims(probabilities) == 3
        n, seq_len, batch_size = size(probabilities)
        𝐚 = zeros(Int, seq_len, batch_size)
        for i in 1:batch_size
            for j in 1:seq_len
                𝐚[j, i] = sample(rng, 1:n, ProbabilityWeights(probabilities[:, j, i]))
            end
        end
    else
        n, batch_size = size(probabilities)
        𝐚 = zeros(Int, batch_size)
        for i in 1:batch_size
            𝐚[i] = sample(rng, 1:n, ProbabilityWeights(probabilities[:, i]))
        end
    end
    return 𝐚
end

function (p::PPOActorDiscrete{T})(𝐬::AbstractArray{Float32}, 𝐚::AbstractArray{Int})::AbstractArray{Float32} where {T}
    get_probs_logprobs(p, 𝐬, 𝐚)[1]
end

function get_probs_logprobs(p::PPOActorDiscrete{T}, 𝐬::AbstractArray{Float32}, 𝐚::AbstractArray{Int})::Tuple{AbstractArray{Float32}, AbstractArray{Float32}} where {T}
    probabilities, logprobabilities = get_probs_logprobs(p, 𝐬)
    @assert ndims(𝐚) == ndims(probabilities) - 1
    if ndims(probabilities) == 3
        seq_len, batch_size = size(𝐚)
        𝐚 = Flux.Zygote.@ignore [CartesianIndex(𝐚[1, t, i], t, i) for t in 1:seq_len, i in 1:batch_size]
    else
        batch_size = length(𝐚)
        𝐚 = Flux.Zygote.@ignore [CartesianIndex(𝐚[i], i) for i in 1:batch_size]
    end
    return probabilities[𝐚], logprobabilities[𝐚]
end

function get_probs_logprobs(p::PPOActorDiscrete{T}, 𝐬::AbstractArray{Float32})::Tuple{AbstractArray{Float32}, AbstractArray{Float32}} where {T}
    if p.recurtype ∈ (MARKOV, TRANSFORMER) || ndims(𝐬) == 2
        logits = p.actor_model(𝐬)
    else
        # interpret as (state_dim, ntimesteps, batch_size) for the RNN
        Flux.Zygote.@ignore Flux.reset!(p.actor_model)
        logits = mapfoldl(hcat, 1:size(𝐬, 2)) do t
            return reshape(p.actor_model(𝐬[:, t, :]), :, 1, size(𝐬, 3))
        end
    end
    if p.deterministic
        logits = logits * 1.0f6
    end
    probabilities = Flux.softmax(logits)
    logprobabilities = Flux.logsoftmax(logits)
    return probabilities, logprobabilities
end

function get_entropy(p::PPOActorDiscrete{T}, 𝐬::AbstractArray{Float32})::Float32 where {T}
    𝛑, log𝛑 = get_probs_logprobs(p, 𝐬)
    return -sum(𝛑 .* log𝛑; dims=1) |> mean
end

function get_entropy(p::PPOActorDiscrete{T}, 𝛑, log𝛑)::Float32 where {T}
    return -sum(𝛑 .* log𝛑; dims=1) |> mean
end

function get_kl_div(p::PPOActorDiscrete, 𝐬, 𝐚, old𝛑, oldlog𝛑)
    𝛑, log𝛑 = get_probs_logprobs(p, 𝐬)
    return sum(old𝛑 .* (oldlog𝛑 .- log𝛑); dims=1) |> mean
end


function get_loss_and_entropy(p::PPOActorDiscrete, 𝐬, 𝐚, 𝛅, old𝛑, oldlog𝛑, ϵ, use_clip_objective=true)
    _, seq_len, batch_size = size(𝐚)
    𝐚 = Flux.Zygote.@ignore [CartesianIndex(𝐚[1, t, i], t, i) for t in 1:seq_len, i in 1:batch_size]
    𝐚 = reshape(𝐚, 1, seq_len, batch_size)
    𝛑, log𝛑 = get_probs_logprobs(p, 𝐬)
    if use_clip_objective
        𝑟 =  𝛑[𝐚] ./ old𝛑[𝐚]
        loss = -min.(𝑟 .* 𝛅, clamp.(𝑟, 1f0-ϵ, 1f0+ϵ) .* 𝛅) |> mean
    else
        loss = -𝛅 .* log𝛑[𝐚] |> mean
    end
    entropy = get_entropy(p, 𝛑, log𝛑)
    return loss, entropy
end







mutable struct PPOActorContinuous{Tₛ <: AbstractFloat, Tₐ <: AbstractFloat} <: AbstractPolicy{Vector{Tₛ}, Vector{Tₐ}}
    const recurtype::RecurrenceType
    actor_model  # maps states to action log probabilities
    deterministic::Bool
    
    logstd::AbstractVector{Float32}
    shift::AbstractVector{Float32}
    scale::AbstractVector{Float32}

    const observation_history::Vector{Vector{Float32}}
    const misc_args
end
function PPOActorContinuous{Tₛ, Tₐ}(recurtype::RecurrenceType, actor_model, deterministic::Bool, aspace::VectorSpace{Tₐ}, misc_args) where {Tₛ, Tₐ}
    n = size(aspace, 1)
    logstd = zeros(n) |> tof32
    shift = (aspace.lows + aspace.highs) / 2  |> tof32
    scale = (aspace.highs - aspace.lows) / 2  |> tof32
    return PPOActorContinuous{Tₛ, Tₐ}(recurtype, actor_model, deterministic, logstd, shift, scale, Vector{Float32}[], misc_args)
end

Flux.@functor PPOActorContinuous (actor_model, logstd)
Flux.gpu(p::PPOActorContinuous{Tₛ, Tₐ}) where {Tₛ, Tₐ}  = PPOActorContinuous{Tₛ, Tₐ}(p.recurtype, Flux.gpu(p.actor_model), p.deterministic, Flux.gpu(p.logstd), Flux.gpu(p.shift), Flux.gpu(p.scale), p.observation_history, p.misc_args)
Flux.cpu(p::PPOActorContinuous{Tₛ, Tₐ}) where {Tₛ, Tₐ}  = PPOActorContinuous{Tₛ, Tₐ}(p.recurtype, Flux.cpu(p.actor_model), p.deterministic, Flux.cpu(p.logstd), Flux.cpu(p.shift), Flux.cpu(p.scale), p.observation_history, p.misc_args)


function (p::PPOActorContinuous{Tₛ, Tₐ})(rng::AbstractRNG, 𝐬::AbstractArray{Float32})::AbstractArray{Float32} where {Tₛ, Tₐ}
    𝐚, log𝛑𝐚 = sample_action_logprobs(p, rng, 𝐬)
    return 𝐚
end

function (p::PPOActorContinuous{Tₛ, Tₐ})(𝐬::AbstractArray{Float32}, 𝐚::AbstractArray{Float32})::AbstractArray{Float32} where {Tₛ, Tₐ}
    get_logprobs(p, 𝐬, 𝐚)
end

function sample_action_logprobs(p::PPOActorContinuous{Tₛ, Tₐ}, rng::AbstractRNG, 𝐬::AbstractArray{Float32})::Tuple{AbstractArray{Float32}, AbstractArray{Float32}} where {Tₛ, Tₐ}
    𝛍 = p.actor_model(𝐬)
    log𝛔 = clamp.(p.logstd, -20f0, 2f0)
    𝛔 = (1f0 - Float32(p.deterministic)) * exp.(log𝛔)
    𝛏 = Flux.Zygote.@ignore convert(typeof(𝛍), randn(rng, Float32, size(𝛍)))
    𝐚 = 𝛍 .+ 𝛔 .* 𝛏
    log𝛑𝐚 = sum(log_nomal_prob.(𝐚, 𝛍, 𝛔), dims=1)
    𝐚 = tanh.(𝐚)
    log𝛑𝐚 = log𝛑𝐚 .- sum(log.(1f0 .- 𝐚 .^ 2 .+ 1f-6), dims=1)
    𝐚 = p.shift .+ p.scale .* 𝐚
    return 𝐚, log𝛑𝐚
end


function get_logprobs(p::PPOActorContinuous{Tₛ, Tₐ}, 𝐬::AbstractArray{Float32}, 𝐚::AbstractArray{Float32})::AbstractArray{Float32} where {Tₛ, Tₐ}
    𝐚 = clamp.(𝐚, -1f0 + 1f-6, 1f0 - 1f-6)
    𝐚_unshifted_unscaled = (𝐚 .- p.shift) ./ p.scale
    𝐚_untanhed = atanh.(𝐚_unshifted_unscaled)

    if p.recurtype ∈ (MARKOV, TRANSFORMER) || ndims(𝐬) == 2
        𝛍 = p.actor_model(𝐬)
    else
        # interpret as (state_dim, ntimesteps, batch_size)
        Flux.Zygote.@ignore Flux.reset!(p.actor_model)
        𝛍 = mapfoldl(hcat, 1:size(𝐬, 2)) do t
            reshape(p.actor_model(𝐬[:, t, :]), 1, 1, size(𝐬, 3))
        end
    end
    𝛍 = p.actor_model(𝐬)
    log𝛔 = clamp.(p.logstd, -20f0, 2f0)
    𝛔 = (1f0 - Float32(p.deterministic)) * exp.(log𝛔)

    log𝛑𝐚 = sum(log_nomal_prob.(𝐚_untanhed, 𝛍, 𝛔), dims=1)
    log𝛑𝐚 = log𝛑𝐚 .- sum(log.(1f0 .- 𝐚_unshifted_unscaled .^ 2 .+ 1f-6), dims=1)

    return log𝛑𝐚
end

function get_entropy(p::PPOActorContinuous)
    D = length(p.logstd)
    logσ = clamp.(p.logstd, -20f0, 2f0)
    return 0.5f0 * D * (1f0 + LOG_2PI) + sum(logσ)
end

function get_entropy(p::PPOActorContinuous, 𝐬)
    get_entropy(p)
end

function get_kl_div(p::PPOActorContinuous, 𝐬, 𝐚, old𝛑, oldlog𝛑)
    log𝛑 = get_logprobs(p, 𝐬, 𝐚)
    logratio = log𝛑 .- oldlog𝛑
    kl = (exp.(logratio) .- 1f0 .- logratio) |> mean  # More stable than -logratio. http://joschu.net/blog/kl-approx.html
    return kl
end


function get_loss_and_entropy(p::PPOActorContinuous, 𝐬, 𝐚, 𝛅, old𝛑, oldlog𝛑, ϵ, use_clip_objective=true)
    log𝛑 = get_logprobs(p, 𝐬, 𝐚)
    𝑟 = exp.(log𝛑 .- oldlog𝛑)
    if use_clip_objective
        loss = -min.(𝑟 .* 𝛅, clamp.(𝑟, 1f0 - ϵ, 1f0 + ϵ) .* 𝛅) |> mean
    else
        loss = 𝛅 .* -log𝛑 |> mean
    end
    entropy = get_entropy(p)
    return loss, entropy
end





const PPOActor{Tₛ, Tₐ} = Union{PPOActorContinuous{Tₛ, Tₐ}, PPOActorDiscrete{Tₛ}}


function MDPs.preepisode(p::PPOActor; env, kwargs...)
    Flux.reset!(p.actor_model)
    empty!(p.observation_history)
    p.recurtype == TRANSFORMER && push!(p.observation_history, tof32(deepcopy(state(env))))
    nothing
end

function (p::PPOActorDiscrete{Tₛ})(rng::AbstractRNG, s::Vector{Tₛ})::Int where {Tₛ}
    ppo_unified(p, rng, s)
end
function (p::PPOActorDiscrete{Tₛ})(s::Vector{Tₛ}, a::Int)::Float64 where {Tₛ}
    ppo_unified(p, s, a)
end

function (p::PPOActorContinuous{Tₛ, Tₐ})(rng::AbstractRNG, s::Vector{Tₛ})::Vector{Tₐ} where {Tₛ, Tₐ}
    ppo_unified(p, rng, s)
end
function (p::PPOActorContinuous{Tₛ, Tₐ})(s::Vector{Tₛ}, a::Vector{Tₐ})::Float64 where {Tₛ, Tₐ}
    ppo_unified(p, s, a)
end

function ppo_unified(p::PPOActor{Tₛ, Tₐ}, rng::AbstractRNG, s::Vector{Tₛ}) where {Tₛ, Tₐ}
    if p.recurtype == TRANSFORMER
        push!(p.observation_history, tof32(deepcopy(s)))
        s = hcat(p.observation_history...)
        # some ugly stuff: TODO: put this into a function or some wrapper.
        # sl = size(s, 2)
        # task_horizon, horizon, extrapolate_to = p.misc_args
        # _CL = decide_context_length(sl, extrapolate_to, horizon, task_horizon)
        # if sl > _CL
        #     if sl % task_horizon == 0
        #         cl = _CL
        #     else
        #         cl = _CL - task_horizon + sl % task_horizon
        #     end
        #     s = s[:, end-cl+1:end]
        #     tstart = s[end, 1]
        #     s[end, :] = s[end, :] .- tstart
        #     s[end, :] = (extrapolate_to/horizon) * s[end, :]
        # end
        # end of ugly stuff.
    end
    𝐬 = s |> batch |> tof32
    a = p(rng, 𝐬) |> unbatch
    if p.recurtype == TRANSFORMER
        a = unbatch_last(a)
    end
    return a
end

function ppo_unified(p::PPOActor{Tₛ, Tₐ}, s::Vector{Tₛ}, a::Union{Int, Vector{Tₐ}})::Float64 where {Tₛ, Tₐ}
    if p.recurtype == TRANSFORMER
        s = hcat(p.observation_history...)
    end
    𝐬 = s |> batch |> tof32
    𝐚 = a |> batch
    πa = p(𝐬, 𝐚) |> unbatch
    if p.recurtype == TRANSFORMER
        πa = unbatch_last(πa)
    end
    return πa
end

