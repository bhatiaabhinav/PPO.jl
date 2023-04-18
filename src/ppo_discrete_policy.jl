using MDPs
using StatsBase
using Flux
using Random

export Actor, RecurrentActor, TransformerActor

struct Actor{T<:AbstractFloat} <: AbstractPolicy{Vector{T}, Int}
    actor_model  # maps states to action log probabilities
    deterministic::Bool
    n::Int
end

Flux.@functor Actor (actor_model, )
Flux.gpu(p::Actor{T}) where {T}  = Actor{T}(Flux.gpu(p.actor_model), p.deterministic, p.n)
Flux.cpu(p::Actor{T}) where {T}  = Actor{T}(Flux.cpu(p.actor_model), p.deterministic, p.n)


function (p::Actor{T})(rng::AbstractRNG, s::Vector{T})::Int where {T}
    𝐬 = reshape(s, :, 1)
    𝐚 = p(rng, 𝐬)
    return 𝐚[1]
    # argmax(p.actor_model(tof32(s)))
end

function (p::Actor{T})(s::Vector{T}, a::Int) where {T}
    𝐬 = reshape(s, :, 1)
    return p(𝐬, :)[a, 1]
end




mutable struct RecurrentActor{T<:AbstractFloat} <: AbstractPolicy{Vector{T}, Int}
    actor_model  # maps evidence to action log probabilities
    deterministic::Bool
    n::Int
    prev_a::Vector{Float32}
    prev_r::Float32
end
function RecurrentActor{T}(actor_model, deterministic, n::Int) where {T}
    RecurrentActor{T}(actor_model, deterministic, n, zeros(Float32, n), 0f0)
end

Flux.@functor RecurrentActor (actor_model, )
Flux.gpu(p::RecurrentActor{T}) where {T}  = RecurrentActor{T}(Flux.gpu(p.actor_model), p.deterministic, p.n, p.prev_a, p.prev_r)
Flux.cpu(p::RecurrentActor{T}) where {T}  = RecurrentActor{T}(Flux.cpu(p.actor_model), p.deterministic, p.n, p.prev_a, p.prev_r)

function MDPs.preepisode(p::RecurrentActor; kwargs...)
    fill!(p.prev_a, 0)
    p.prev_r = 0
    Flux.reset!(p.actor_model)
    nothing
end

function (p::RecurrentActor{T})(rng::AbstractRNG, o::Vector{T})::Int where {T}
    𝐬 = reshape(vcat(p.prev_a, p.prev_r, tof32(o)), :, 1)
    𝐚 = p(rng, 𝐬)
    return 𝐚[1]
end

function (p::RecurrentActor{T})(o::Vector{T}, a::Int) where {T}
    𝐬 = reshape(vcat(p.prev_a, p.prev_r, tof32(o)), :, 1)
    return p(𝐬, :)[a, 1]
end

function MDPs.poststep(p::RecurrentActor{T}; env, kwargs...) where {T}
    fill!(p.prev_a, 0f0)
    p.prev_a[action(env)] = 1f0
    p.prev_r = reward(env)
    # @debug "Storing action, reward in policy struct"
    nothing
end 



const SomeActor{T} = Union{Actor{T}, RecurrentActor{T}}


function (p::SomeActor{T})(rng::AbstractRNG, 𝐬::AbstractMatrix{<:AbstractFloat})::Vector{Int} where {T}
    𝐬 = tof32(𝐬)
    probabilities = p(𝐬, :)
    n, batch_size = size(probabilities)
    π = zeros(Int, batch_size)
    for i in 1:batch_size
        π[i] = sample(rng, 1:n, ProbabilityWeights(probabilities[:, i]))
    end
    return π
end

function (p::SomeActor{T})(𝐬::AbstractMatrix{<:AbstractFloat}, 𝐚::AbstractVector{Int})::AbstractVector{Float32} where {T}
    probabilities = p(𝐬, :)
    batch_size = length(𝐚)
    action_indices = [CartesianIndex(𝐚[i], i) for i in 1:batch_size]
    return probabilities[action_indices]
end


function (p::SomeActor{T})(𝐬::AbstractMatrix{<:AbstractFloat}, ::Colon)::AbstractMatrix{Float32} where {T}
    𝐬 = tof32(𝐬)
    logits = p.actor_model(𝐬)
    if p.deterministic
        logits = logits * 1.0f6
    end
    probabilities = Flux.softmax(logits)
    return probabilities
end

function get_probs_logprobs(p::SomeActor{T}, 𝐬::AbstractMatrix{<:AbstractFloat})::Tuple{AbstractMatrix{Float32}, AbstractMatrix{Float32}} where {T}
    𝐬 = tof32(𝐬)
    logits = p.actor_model(𝐬)
    if p.deterministic
        logits = logits * 1.0f6
    end
    probabilities = Flux.softmax(logits)
    logprobabilities = Flux.logsoftmax(logits)
    return probabilities, logprobabilities
end



mutable struct TransformerActor{T <: AbstractFloat} <: AbstractPolicy{Vector{T}, Int}
    transformer
    deterministic::Bool
    n::Int
    prev_a::Vector{Float32}
    prev_r::Float32
    prev_d::Float32
    evidence_seq
end
function TransformerActor{T}(transformer, deterministic, n::Int) where {T}
    TransformerActor{T}(transformer, deterministic, n, zeros(Float32, n), 0f0, 1f0 ,nothing)
end

Flux.@functor TransformerActor (transformer, )
Flux.gpu(p::TransformerActor{T}) where {T}  = TransformerActor{T}(Flux.gpu(p.transformer), p.deterministic, p.n, p.prev_a, p.prev_r, p.prev_d, p.evidence_seq)
Flux.cpu(p::TransformerActor{T}) where {T}  = TransformerActor{T}(Flux.cpu(p.transformer), p.deterministic, p.n, p.prev_a, p.prev_r, p.prev_d, p.evidence_seq)

function MDPs.preepisode(p::TransformerActor{T}; env, kwargs...) where {T}
    fill!(p.prev_a, 0)
    p.prev_r = 0
    p.prev_d = 1
    o::Vector{T} = state(env)
    p.evidence_seq = reshape(vcat(p.prev_a, p.prev_r, p.prev_d, tof32(o)), :, 1)
    nothing
end

function (p::TransformerActor{T})(rng::AbstractRNG, o::Vector{T})::Int where {T}
    𝐬 = reshape(vcat(p.prev_a, p.prev_r, p.prev_d,  tof32(o)), :, 1)
    if p.evidence_seq === nothing
        evidence_seq = 𝐬
    else
        evidence_seq = @views hcat(p.evidence_seq[:, 1:end-1], 𝐬)
    end
    evidence_seq_batched = reshape(evidence_seq, size(evidence_seq)..., 1)
    𝐚 = p(rng, evidence_seq_batched) # returns an action for each timestep for each batch
    return 𝐚[end, 1]
end

function (p::TransformerActor{T})(o::Vector{T}, a::Int) where {T}
    𝐬 = reshape(vcat(p.prev_a, p.prev_r, p.prev_d, tof32(o)), :, 1)
    if p.evidence_seq === nothing
        evidence_seq = 𝐬
    else
        evidence_seq = @views hcat(p.evidence_seq[:, end-1], 𝐬)
    end
    evidence_seq_batched = reshape(evidence_seq, size(evidence_seq)..., 1)
    return p(evidence_seq_batched, :)[a, end, 1]
end

function MDPs.poststep(p::TransformerActor{T}; env, kwargs...) where {T}
    fill!(p.prev_a, 0f0)
    p.prev_a[action(env)] = 1f0
    p.prev_r = reward(env)
    p.prev_d = in_absorbing_state(env) |> Float32
    o::Vector{T} = state(env)
    𝐬 = reshape(vcat(p.prev_a, p.prev_r, p.prev_d, tof32(o)), :, 1)
    p.evidence_seq = hcat(p.evidence_seq, 𝐬)
    nothing
end

function (p::TransformerActor{T})(rng::AbstractRNG, evidence_seq::AbstractArray{<:AbstractFloat, 3})::Matrix{Int} where {T}
    evidence_seq = tof32(evidence_seq)
    probabilities = p(evidence_seq, :)  # returns a probability for each action for each timestep for each batch. (n_actions, n_timesteps, n_batches)
    n, seq_len, batch_size = size(probabilities)
    π = zeros(Int, seq_len, batch_size)
    for i in 1:batch_size
        for j in 1:seq_len
            π[j, i] = sample(rng, 1:n, ProbabilityWeights(probabilities[:, j, i]))
        end
    end
    return π
end


function (p::TransformerActor{T})(evidence_seq::AbstractArray{<:AbstractFloat, 3}, 𝐚::AbstractMatrix{Int})::AbstractMatrix{Float32} where {T}
    evidence_seq = tof32(evidence_seq)
    probabilities = p(evidence_seq, :) # returns a probability for each action for each timestep for each batch. (n_actions, n_timesteps, n_batches)
    seq_len, batch_size = size(𝐚)
    action_indices = [CartesianIndex(𝐚[j, i], j, i) for j in 1:seq_len, i in 1:batch_size]
    return probabilities[action_indices]
end


function (p::TransformerActor{T})(evidence_seq::AbstractArray{<:AbstractFloat, 3}, ::Colon)::AbstractArray{Float32, 3} where {T}
    evidence_seq = tof32(evidence_seq)
    logits = p.transformer(evidence_seq) # returns logits for each action for each timestep for each batch. (n_actions, n_timesteps, n_batches)
    if p.deterministic
        logits = logits * 1.0f6
    end
    probabilities = Flux.softmax(logits)
    return probabilities
end


function get_probs_logprobs(p::TransformerActor{T}, evidence_seq::AbstractArray{<:AbstractFloat, 3})::Tuple{AbstractArray{Float32, 3}, AbstractArray{Float32, 3}} where {T}
    evidence_seq = tof32(evidence_seq)
    logits = p.transformer(evidence_seq) # returns logits for each action for each timestep for each batch. (n_actions, n_timesteps, n_batches)
    if p.deterministic
        logits = logits * 1.0f6
    end
    probabilities = Flux.softmax(logits)
    logprobabilities = Flux.logsoftmax(logits)
    return probabilities, logprobabilities
end