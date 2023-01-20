using MDPs
using StatsBase
using Flux
using Random

export Actor, RecurrentActor

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
    Flux.reset!(p.actor_model)
end

function (p::RecurrentActor{T})(rng::AbstractRNG, o::Vector{T})::Int where {T}
    𝐬 = reshape(vcat(p.prev_a, p.prev_r, tof32(o), 0f0), :, 1)
    𝐚 = p(rng, 𝐬)
    return 𝐚[1]
end

function (p::RecurrentActor{T})(o::Vector{T}, a::Int) where {T}
    𝐬 = reshape(vcat(p.prev_a, p.prev_r, tof32(o), 0f0), :, 1)
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
    π = zeros(Int, n)
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


