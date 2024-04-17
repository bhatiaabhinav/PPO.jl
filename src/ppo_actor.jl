using MDPs
using StatsBase
using Flux
using Random

export PPOActorDiscrete, PPOActorContinuous, RecurrenceType, MARKOV, RECURRENT, TRANSFORMER

@enum RecurrenceType MARKOV RECURRENT TRANSFORMER


mutable struct PPOActorDiscrete{T<:AbstractFloat} <: AbstractPolicy{Vector{T}, Int}
    const recurtype::RecurrenceType
    actor_model  # maps states to action logits
    deterministic::Bool
    const n::Int # number of actions

    observation_history::Union{AbstractMatrix{Float32}, Nothing}
    obs_history_len::Int
    const device
end

"""
    PPOActorDiscrete{T}(actor_model, deterministic::Bool, aspace::IntegerSpace, recurtype::RecurrenceType=MARKOV) where {T}

Construct a PPOActorDiscrete policy, where the states are of type Vector{T}. The policy is parameterized by a Flux model, which can be recurrent or a transformer to handle sequential inputs. The policy can be stochastic or deterministic.

# Arguments
- `actor_model`: a Flux model that maps states to action logits.
- `deterministic`: whether the policy is deterministic or stochastic. If deterministic, the policy will always output the action with the highest probability. If stochastic, the policy will sample actions from the probability distribution over actions.
- `aspace`: the action space
- `recurtype`: the recurrence type of the neural network (MARKOV, RECURRENT, TRANSFORMER). Defaults to MARKOV (no recurrence).
"""
function PPOActorDiscrete{T}(actor_model, deterministic::Bool, aspace::IntegerSpace, recurtype::RecurrenceType=MARKOV) where {T}
    device = isa(first(Flux.params(actor_model)), Array) ? Flux.cpu : Flux.gpu
    return PPOActorDiscrete{T}(recurtype, actor_model, deterministic, length(aspace), nothing, 0, device)
end

Flux.@functor PPOActorDiscrete (actor_model, )
Flux.gpu(p::PPOActorDiscrete{T}) where {T}  = PPOActorDiscrete{T}(p.recurtype, Flux.gpu(p.actor_model), p.deterministic, p.n, p.observation_history, p.obs_history_len, Flux.gpu)
Flux.cpu(p::PPOActorDiscrete{T}) where {T}  = PPOActorDiscrete{T}(p.recurtype, Flux.cpu(p.actor_model), p.deterministic, p.n, p.observation_history, p.obs_history_len, Flux.cpu)

"""
    (p::PPOActorDiscrete{T})(rng::AbstractRNG, 𝐬::AbstractArray{Float32})::VecOrMat{Int} where {T}

Sample actions from the policy given the states 𝐬. States 𝐬 are assumed to be in the form (state_dim, ntimesteps, batch_size) or (state_dim, batch_size).
# Returns
- `𝐚`: the sampled actions as a Vector{Int} for markov policies and a Matrix{Int} for recurrent policies
"""
function (p::PPOActorDiscrete{T})(rng::AbstractRNG, 𝐬::AbstractArray{Float32})::VecOrMat{Int} where {T}
    probabilities, logprobabilities = get_probs_logprobs(p, 𝐬)
    probabilities = probabilities |> Flux.cpu
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
            𝐚[i] = sample(rng, 1:n, ProbabilityWeights(@view probabilities[:, i]))
        end
    end
    return 𝐚
end

"""
    (p::PPOActorDiscrete{T})(𝐬::AbstractArray{Float32}, 𝐚::AbstractArray{Int})::AbstractArray{Float32} where {T}

Get the probabilities of the actions 𝐚 given the states 𝐬. States 𝐬 are assumed to be in the form (state_dim, ntimesteps, batch_size) or (state_dim, batch_size). Actions are accordingly assumed to be in the form (ntimesteps, batch_size) or (batch_size).

# Returns
- `probabilities`: the probabilities of the actions 𝐚 given the states 𝐬. 

The output is of the same shape as 𝐚.
"""
function (p::PPOActorDiscrete{T})(𝐬::AbstractArray{Float32}, 𝐚::AbstractArray{Int})::AbstractArray{Float32} where {T}
    get_probs_logprobs(p, 𝐬, 𝐚)[1]
end

"""
    get_probs_logprobs(p::PPOActorDiscrete, 𝐬::AbstractArray{Float32}, 𝐚::AbstractArray{Int})::Tuple{AbstractArray{Float32}, AbstractArray{Float32}}

Get the probabilities and log probabilities of the actions 𝐚 given the states 𝐬. States 𝐬 are assumed to be in the form (state_dim, ntimesteps, batch_size) or (state_dim, batch_size). Actions are accordingly assumed to be in the form (ntimesteps, batch_size) or (batch_size).

# Returns
- `probabilities`: the probabilities of the actions 𝐚 given the states 𝐬.
- `logprobabilities`: the log probabilities of the actions 𝐚 given the states 𝐬

The outputs are of the same shape as 𝐚.
"""
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

"""
    get_probs_logprobs(p::PPOActorDiscrete, 𝐬::AbstractArray{Float32})::Tuple{AbstractArray{Float32}, AbstractArray{Float32}}

Get the probabilities and log probabilities of all actions given the states 𝐬. States 𝐬 are assumed to be in the form (state_dim, ntimesteps, batch_size) or (state_dim, batch_size).

# Returns
- `probabilities`: the probabilities of all actions given the states 𝐬.
- `logprobabilities`: the log probabilities of all actions given the states 𝐬

The outputs are of the shape (n, ntimesteps, batch_size) or (n, batch_size) depending on the shape of 𝐬, where n is the number of actions.
"""
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

"""
    get_entropy(p::PPOActorDiscrete, 𝐬::AbstractArray{Float32})::Float32 where {T}

Get the entropy of the policy given the states 𝐬
"""
function get_entropy(p::PPOActorDiscrete{T}, 𝐬::AbstractArray{Float32})::AbstractArray{Float32} where {T}
    𝛑, log𝛑 = get_probs_logprobs(p, 𝐬)
    return -sum(𝛑 .* log𝛑; dims=1)
end

"""
    get_entropy(p::PPOActorDiscrete{T}, 𝛑, log𝛑)::Float32 where {T}

Get the entropy given the probabilities 𝛑 and log probabilities log𝛑.
"""
function get_entropy(p::PPOActorDiscrete{T}, 𝛑, log𝛑)::AbstractArray{Float32} where {T}
    return -sum(𝛑 .* log𝛑; dims=1)
end

"""
    get_kl_div(p::PPOActorDiscrete, 𝐬, 𝐚, old𝛑, oldlog𝛑)

Get the KL divergence between the old policy and the current policy given the states 𝐬, actions 𝐚, and old probabilities old𝛑 and old log probabilities oldlog𝛑 for actions 𝐚 in states 𝐬
"""
function get_kl_div(p::PPOActorDiscrete, 𝐬, 𝐚, old𝛑, oldlog𝛑)
    𝛑, log𝛑 = get_probs_logprobs(p, 𝐬)
    return sum(old𝛑 .* (oldlog𝛑 .- log𝛑); dims=1) |> mean
end


function get_loss_and_entropy(p::PPOActorDiscrete, 𝐬, 𝐚, 𝛅, old𝛑, oldlog𝛑, ϵ, use_clip_objective=true)
    if ndims(𝐬) == 3
        _, seq_len, batch_size = size(𝐚)
        𝐚 = Flux.Zygote.@ignore [CartesianIndex(𝐚[1, t, i], t, i) for t in 1:seq_len, i in 1:batch_size]
        𝐚 = reshape(𝐚, 1, seq_len, batch_size)
    else
        _, batch_size = size(𝐚)
        𝐚 = Flux.Zygote.@ignore [CartesianIndex(𝐚[1, i], i) for i in 1:batch_size]
        𝐚 = reshape(𝐚, 1, batch_size)
    end
    𝛑, log𝛑 = get_probs_logprobs(p, 𝐬)
    if use_clip_objective
        𝑟 =  𝛑[𝐚] ./ old𝛑[𝐚]
        loss = -min.(𝑟 .* 𝛅, clamp.(𝑟, 1f0-ϵ, 1f0+ϵ) .* 𝛅) |> mean
    else
        loss = -𝛅 .* log𝛑[𝐚] |> mean
    end
    entropy = get_entropy(p, 𝛑, log𝛑) |> mean
    return loss, entropy
end







mutable struct PPOActorContinuous{Tₛ <: AbstractFloat, Tₐ <: AbstractFloat} <: AbstractPolicy{Vector{Tₛ}, Vector{Tₐ}}
    const recurtype::RecurrenceType
    actor_model  # maps states to mean of action distribution
    deterministic::Bool
    
    const state_dependent_noise::Bool
    logstd::AbstractVector{Float32}
    shift::AbstractVector{Float32}
    scale::AbstractVector{Float32}

    observation_history::Union{AbstractMatrix{Float32}, Nothing}
    obs_history_len::Int
    const device
end

"""
    PPOActorContinuous{Tₛ, Tₐ}(actor_model, deterministic::Bool, aspace::VectorSpace{Tₐ}, recurtype::RecurrenceType=MARKOV) where {Tₛ, Tₐ}

Create a continuous actor for PPO with a continuous action space. The states and actions are of type `Vector{Tₛ}` and `Vector{Tₐ}` respectively. The actor model maps states to the mean of the action distribution, and can be a recurrent neural network or a transformer to handle sequential inputs. The standard deviation of the action distribution is a learnable parameter. The action is sampled from a normal distribution with the mean and standard deviation. The action is then squashed using the tanh function and scaled and shifted to fit the action space. # TODO: Describe the state-dependent noise case.

# Arguments
- `actor_model`: the Flux model
- `deterministic`: whether to sample actions from the distribution (deterministic=false) or simply take the mean of the distribution (deterministic=true).
- `aspace`: the action space
- `recurtype`: the recurrence type, either `MARKOV`, `RECURRENT`, or `TRANSFORMER`. Defaults to `MARKOV` (no recurrence).
- `state_dependent_noise=false`: whether the standard deviation of the action distribution is state-dependent. If true, the standard deviation is a function of the state. In that case, the actor model must return 2n outputs, where n is the dimension of the action space. The first n outputs are the mean of the action distribution, and the last n outputs are the log standard deviation of the action distribution. If false, the standard deviation is a learnable parameter that is independent of the state.
"""
function PPOActorContinuous{Tₛ, Tₐ}(actor_model, deterministic::Bool, aspace::VectorSpace{Tₐ}, recurtype::RecurrenceType=MARKOV; state_dependent_noise=false) where {Tₛ, Tₐ}
    n = size(aspace, 1)
    logstd = zeros(n) |> tof32
    shift = (aspace.lows + aspace.highs) / 2  |> tof32
    scale = (aspace.highs - aspace.lows) / 2  |> tof32
    device = isa(first(Flux.params(actor_model)), Array) ? Flux.cpu : Flux.gpu
    return PPOActorContinuous{Tₛ, Tₐ}(recurtype, actor_model, deterministic, state_dependent_noise, device(logstd), device(shift), device(scale), nothing, 0, device)
end

Flux.@functor PPOActorContinuous (actor_model, logstd)
Flux.gpu(p::PPOActorContinuous{Tₛ, Tₐ}) where {Tₛ, Tₐ}  = PPOActorContinuous{Tₛ, Tₐ}(p.recurtype, Flux.gpu(p.actor_model), p.deterministic, p.state_dependent_noise,  Flux.gpu(p.logstd), Flux.gpu(p.shift), Flux.gpu(p.scale), p.observation_history, p.obs_history_len, Flux.gpu)
Flux.cpu(p::PPOActorContinuous{Tₛ, Tₐ}) where {Tₛ, Tₐ}  = PPOActorContinuous{Tₛ, Tₐ}(p.recurtype, Flux.cpu(p.actor_model), p.deterministic, p.state_dependent_noise, Flux.cpu(p.logstd), Flux.cpu(p.shift), Flux.cpu(p.scale), p.observation_history, p.obs_history_len, Flux.cpu)

"""
    (p::PPOActorContinuous{Tₛ, Tₐ})(rng::AbstractRNG, 𝐬::AbstractArray{Float32})::AbstractArray{Float32} where {Tₛ, Tₐ}

Given states 𝐬, samples and returns actions. If input is of shape (state_dims, ntimesteps, batch_size), then the output is of shape (action_dims, ntimesteps, batch_size). If input is of shape (state_dims, batch_size), then the output is of shape (action_dims, batch_size).
"""
function (p::PPOActorContinuous{Tₛ, Tₐ})(rng::AbstractRNG, 𝐬::AbstractArray{Float32})::AbstractArray{Float32} where {Tₛ, Tₐ}
    𝐚, log𝛑𝐚 = sample_action_logprobs(p, rng, 𝐬)
    return 𝐚
end

function (p::PPOActorContinuous{Tₛ, Tₐ})(𝐬::AbstractArray{Float32}, 𝐚::AbstractArray{Float32})::AbstractArray{Float32} where {Tₛ, Tₐ}
    get_logprobs(p, 𝐬, 𝐚)
end

function get_mean_logstd(p::PPOActorContinuous{Tₛ, Tₐ}, 𝐬::AbstractArray{Float32})::Tuple{AbstractArray{Float32}, AbstractArray{Float32}} where {Tₛ, Tₐ}
    if p.recurtype ∈ (MARKOV, TRANSFORMER) || ndims(𝐬) == 2
        𝛍 = p.actor_model(𝐬)
    else
        # interpret as (state_dim, ntimesteps, batch_size)
        Flux.Zygote.@ignore Flux.reset!(p.actor_model)
        𝛍 = mapfoldl(hcat, 1:size(𝐬, 2)) do t
            reshape(p.actor_model(𝐬[:, t, :]), :, 1, size(𝐬, 3))
        end
    end
    logstd = p.logstd
    if p.state_dependent_noise
        n::Int = length(p.scale)
        𝛍, logstd = copy(selectdim(𝛍, 1, 1:n)), copy(selectdim(𝛍, 1, n+1:2n))
    end
    log𝛔 = clamp.(logstd, -20f0, 2f0)
    return 𝛍, log𝛔
end

"""
    sample_action_logprobs(p::PPOActorContinuous{Tₛ, Tₐ}, rng::AbstractRNG, 𝐬::AbstractArray{Float32}; return_logstd=false) where {Tₛ, Tₐ}

Given states 𝐬, samples and returns actions and their log probabilities. If input is of shape (state_dims, ntimesteps, batch_size), then outputs are of shape (action_dims, ntimesteps, batch_size) and (1, nsteps, batch_size) respectively. If input is of shape (state_dims, batch_size), then outputs are of shape (action_dims, batch_size) and (1, batch_size) respectively. If return_logstd is true, then the log standard deviation of the action distribution (before squashing) is also returned.
"""
function sample_action_logprobs(p::PPOActorContinuous{Tₛ, Tₐ}, rng::AbstractRNG, 𝐬::AbstractArray{Float32}; return_logstd=false) where {Tₛ, Tₐ}
    𝛍, log𝛔 = get_mean_logstd(p, 𝐬)
    𝛔 = (1f0 - Float32(p.deterministic)) * exp.(log𝛔)
    𝛏 = Flux.Zygote.@ignore convert(typeof(𝛍), randn(rng, Float32, size(𝛍)))
    𝐚 = 𝛍 .+ 𝛔 .* 𝛏
    log𝛑𝐚 = sum(log_nomal_prob.(𝐚, 𝛍, 𝛔), dims=1)
    𝐚 = tanh.(𝐚)
    log𝛑𝐚 = log𝛑𝐚 .- sum(log.(1f0 .- 𝐚 .^ 2 .+ 1f-6), dims=1)
    𝐚 = p.shift .+ p.scale .* 𝐚
    return return_logstd ? (𝐚, log𝛑𝐚, log𝛔) : (𝐚, log𝛑𝐚)
end

"""
    get_logprobs(p::PPOActorContinuous{Tₛ, Tₐ}, 𝐬::AbstractArray{Float32}, 𝐚::AbstractArray{Float32}; return_logstd=false) where {Tₛ, Tₐ}

Given states 𝐬 and actions 𝐚, returns log probabilities of actions. If input is of shape (state_dims, ntimesteps, batch_size), then outputs are of shape (1, nsteps, batch_size). If input is of shape (state_dims, batch_size), then outputs are of shape (1, batch_size). If return_logstd is true, then the log standard deviation of the action distribution (before squashing) is also returned.
"""
function get_logprobs(p::PPOActorContinuous{Tₛ, Tₐ}, 𝐬::AbstractArray{Float32}, 𝐚::AbstractArray{Float32}; return_logstd=false) where {Tₛ, Tₐ}
    @assert all(isfinite.(𝐚))
    𝐚_unshifted_unscaled = (𝐚 .- p.shift) ./ (p.scale .+ 1f-6)
    𝐚_unshifted_unscaled = clamp.(𝐚_unshifted_unscaled, -1f0 + 1f-3, 1f0 - 1f-3)    # because atanh(1.0) is infinite
    @assert all(isfinite.(𝐚_unshifted_unscaled))
    𝐚_untanhed = atanh.(𝐚_unshifted_unscaled)
    @assert all(isfinite.(𝐚_untanhed))

    𝛍, log𝛔 = get_mean_logstd(p, 𝐬)
    𝛔 = (1f0 - Float32(p.deterministic)) * exp.(log𝛔)
    log𝛑𝐚 = sum(log_nomal_prob.(𝐚_untanhed, 𝛍, 𝛔), dims=1)
    log𝛑𝐚 = log𝛑𝐚 .- sum(log.(1f0 .- 𝐚_unshifted_unscaled .^ 2 .+ 1f-6), dims=1)

    return return_logstd ? (log𝛑𝐚, log𝛔) : log𝛑𝐚
end

"""
    get_gaussian_entropy(logσ::AbstractArray{Float32})::AbstractArray{Float32} where {Tₛ, Tₐ}
Returns entropies given logσ.
"""
function get_gaussian_entropy(logσ::AbstractArray{Float32})::AbstractArray{Float32}
    return sum(0.5f0 * (1f0 + LOG_2PI) .+ logσ, dims=1)
end

"""
    get_entropy(p::PPOActorContinuous{Tₛ, Tₐ}, 𝐬::AbstractArray{Float32})::AbstractArray{Float32} where {Tₛ, Tₐ}
Returns entropy of the policy for states 𝐬, without accounting for the squashing. Returns 1-element vector if state_dependent_noise is false, otherwise if input is of shape (state_dims, ntimesteps, batch_size), then outputs are of shape (1, nsteps, batch_size). If input is of shape (state_dims, batch_size), then outputs are of shape (1, batch_size).
"""
function get_entropy(p::PPOActorContinuous, 𝐬::AbstractArray{Float32})::AbstractArray{Float32}
    _, log𝛔 = get_mean_logstd(p, 𝐬)
    return get_gaussian_entropy(log𝛔)
end

"""
    get_kl_div(p::PPOActorContinuous, 𝐬, 𝐚, old𝛑, oldlog𝛑)

Returns KL divergence between old policy and current policy given states 𝐬, actions 𝐚, and old probabilities and log probabilities of taking those actions under the old policy.
"""
function get_kl_div(p::PPOActorContinuous, 𝐬, 𝐚, old𝛑, oldlog𝛑)
    log𝛑 = get_logprobs(p, 𝐬, 𝐚)
    logratio = log𝛑 .- oldlog𝛑
    kl = (exp.(logratio) .- 1f0 .- logratio) |> mean  # More stable than -logratio. http://joschu.net/blog/kl-approx.html
    return kl
end


function get_loss_and_entropy(p::PPOActorContinuous, 𝐬, 𝐚, 𝛅, old𝛑, oldlog𝛑, ϵ, use_clip_objective=true)
    log𝛑, log𝛔 = get_logprobs(p, 𝐬, 𝐚; return_logstd=true)
    𝑟 = exp.(log𝛑 .- oldlog𝛑)
    if use_clip_objective
        loss = -min.(𝑟 .* 𝛅, clamp.(𝑟, 1f0 - ϵ, 1f0 + ϵ) .* 𝛅) |> mean
    else
        loss = 𝛅 .* -log𝛑 |> mean
    end
    entropy = get_gaussian_entropy(log𝛔) |> mean
    return loss, entropy
end





const PPOActor{Tₛ, Tₐ} = Union{PPOActorContinuous{Tₛ, Tₐ}, PPOActorDiscrete{Tₛ}}


function MDPs.preepisode(p::PPOActor; env, kwargs...)
    Flux.reset!(p.actor_model)
    if p.recurtype == TRANSFORMER
        s = state(env)
        if isnothing(p.observation_history)
            p.observation_history = zeros(Float32, length(s), 2) |> p.device
        end
        p.obs_history_len = 1
        p.observation_history[:, 1] = s
    end
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
        if size(p.observation_history, 2) == p.obs_history_len
            new_buffer = zeros(Float32, length(s), 2 * p.obs_history_len) |> p.device
            new_buffer[:, 1:p.obs_history_len] = p.observation_history
            p.observation_history = new_buffer
        end
        p.obs_history_len += 1
        p.observation_history[:, p.obs_history_len] = s
        𝐬 = p.observation_history[:, 1:p.obs_history_len] |> batch
        a = p(rng, 𝐬) |> cpu |> unbatch |> unbatch_last
    else
        𝐬 = s |> batch |> tof32 |> p.device
        a = p(rng, 𝐬) |> cpu |> unbatch
    end
    return a
end

function ppo_unified(p::PPOActor{Tₛ, Tₐ}, s::Vector{Tₛ}, a::Union{Int, Vector{Tₐ}})::Float64 where {Tₛ, Tₐ}
    if p.recurtype == TRANSFORMER
        s = p.observation_history[:, 1:p.obs_history_len]
    end
    𝐬 = s |> batch |> tof32 |> p.device
    𝐚 = a |> batch |> p.device
    πa = p(𝐬, 𝐚) |> Flux.cpu |> unbatch
    if p.recurtype == TRANSFORMER
        πa = unbatch_last(πa)
    end
    return πa
end

