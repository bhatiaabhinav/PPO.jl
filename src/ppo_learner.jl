using MDPs
import MDPs: preexperiment, postepisode, poststep
using Base.Iterators: partition, product
using Random
using Flux
using Flux.Zygote
import ProgressMeter: @showprogress, Progress, next!, finish!

export PPOLearner

"""
    PPOLearner(; envs, actor, critic, γ=0.99, nsteps=2048, nepochs=10, batch_size=64, entropy_bonus=0.0, decay_ent_bonus=false, normalize_advantages=true, clipnorm=0.5, adam_weight_decay=0.0, adam_epsilon=1e-7, lr_actor=0.0003, lr_critic=0.0003, decay_lr=false, min_lr=1.25e-5, λ=0.95, ϵ=0.2, kl_target=Inf, ppo=true, early_stop_critic=false, device=cpu, progressmeter=false, iters_per_postepisode::Int=1)

A hook that performs an iteration of Proximal Policy Optimization (PPO) in `postepisode` callback. Default hyperparameters are similar to those in Stable Baselines3 PPO implementation (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

# Arguments
- `envs::AbstractVecEnv`: A collection of environments to collect data from. Should be of type `AbstractVecEnv`. See `VecEnv` in MDPs.jl to convert an array of environments to a single vectorized environment.
- `actor`: A PPO policy to optimize. Either PPOActorDiscrete or PPOActorContinuous.
- `critic`: A Flux model with recurrence type same as actor.
- `γ::Float32=0.99`: Discount factor. Used to calulate TD(λ) advantages.
- `nsteps::Int=2048`: Numer of steps per iteration. So that total data per iteration = nenvs * nsteps. With longer nsteps, TD(λ) returns are computed better.
- `nepochs::Int=10`: Number of epochs per iteration.
- `batch_size::Int=64`: Minibatch size. Should be a multiple of `nsteps` when using recurrent or transformer actor.
- `entropy_bonus::Float32=0.0`: Coefficient of the entropy term in the overall PPO loss, to encourage exploration.
- `entropy_method::Symbol=:regularized`: `:regularized` or `:maximized`. If `:regularized`, entropy bonus is added to the loss. If `:maximized`, entropy bonus is added to the reward at each timestep (as in SAC).
- `decay_ent_bonus::Bool=false`: Whether to decay entropy bonus over time to 0, by the end of training (after `max_trials` iterations).
- `normalize_advantages::Bool=true`: Whether to center and scale advantages to have zero mean and unit variance
- `clipnorm::Float32=0.5`: Clip gradients by global norm
- `adam_weight_decay::Float32=0.0`: Adam weight decay
- `adam_epsilon::Float32=1e-7`: Adam epsilon
- `lr_actor::Float32=0.0003`: Adam learning rate for actor
- `lr_critic::Float32=0.0003`: Adam learning rate for critic
- `decay_lr::Bool=false`: Whether to decay learning rate over time to 0, by the end of training (after `max_trials` iterations).
- `min_lr::Float32=1.25e-5`: Minimum learning rate to decay to
- `λ::Float32=0.95`: Used to calulate TD(λ) advantages using Generalized Advantage Estimation (GAE) method.
- `ϵ::Float32=0.2`: Epsilon used in PPO clip objective
- `kl_target=Inf`: In each iteration, early stop actor training if KL divergence from old policy exceeds this value. Disbled if `kl_target=Inf` (default). A commonly used value, when enabled, is 0.01.
- `ppo=true`: Whether to use PPO clip objective. If false, standard advantange actor-critic (A2C) objective will be used.
- `early_stop_critic=false`: Whether to early stop training critic (along with actor) if KL divergence from old policy exceeds `kl_target`.
- `device=cpu`: `cpu` or `gpu`
- `progressmeter=false`: Whether to show data and gradient updates progress using a progressmeter (useful for debugging).
- `iters_per_postepisode::Int=1`: Number of training iterations per postepisode callback. Useful for debugging.
"""
Base.@kwdef mutable struct PPOLearner <: AbstractHook
    envs::AbstractVecEnv        # a vectorized environment
    actor::PPOActor
    critic                      # some model with recurrence type same as actor
    γ::Float32 = 0.99           # discount factor. Used to calulate TD(λ) advantages.
    nsteps::Int = 2048           # numer of steps per iteration. So that total data per iteration = nenvs * nsteps. With longer nsteps, TD(λ) returns are computed better.
    nepochs::Int = 10           # number of epochs per iteration.
    batch_size::Int = 64        # minibatch size
    entropy_bonus::Float32 = 0.0f0 # coefficient of the entropy term in the overall PPO loss, to encourage exploration.
    entropy_method::Symbol = :regularized  # :regularized, :maximized
    decay_ent_bonus::Bool = false # whether to decay entropy bonus
    normalize_advantages::Bool = true # whether to center and scale advantages to have zero mean and unit variance
    clipnorm::Float32 = 0.5     # clip gradients by global norm
    adam_weight_decay::Float32 = 0f0      # adam weight decay
    adam_epsilon::Float32 = 1f-7    # adam epsilon
    lr_actor::Float32 = 0.0003        # adam learning rate for actor
    lr_critic::Float32 = 0.0003        # adam learning rate for critic
    decay_lr::Bool = false      # whether to decay learning rate
    min_lr::Float32 = 1.25e-5   # minimum learning rate to decay to
    λ::Float32 = 0.95f0                  # Used to calulate TD(λ) advantages using Generalized Advantage Estimation (GAE) method.
    ϵ::Float32 = 0.2f0                   # epsilon used in PPO clip objective
    kl_target = Inf            # In each iteration, early stop training if KL divergence from old policy exceeds this value.
    ppo = true                  # whether to use PPO clip objective. If false, standard advantange actor-critic (A2C) objective will be used.
    early_stop_critic = false
    device = cpu                # `cpu` or `gpu`
    progressmeter::Bool = false # Whether to show data and gradient updates progress using a progressmeter
    iters_per_postepisode::Int = 1  # number of training iterations per postepisode callback

    # data structures:
    optim_actor = make_adam_optim(lr_actor, (0.9, 0.999), adam_epsilon, 0)
    optim_critic = make_adam_optim(lr_critic, (0.9, 0.999), adam_epsilon, adam_weight_decay)  # regularize critic with weight decay (l2 norm) but don't regularize actor
    actor_gpu = device(deepcopy(actor))
    critic_gpu = device(deepcopy(critic))

    stats = Dict{Symbol, Any}()
end

function preexperiment(ppo::PPOLearner; rng, kwargs...)
    reset!(ppo.envs; rng=rng)
    ppo.stats[:iterations] = 0
end

function postepisode(ppo::PPOLearner; returns, max_trials, rng, kwargs...)
    M, N = ppo.nsteps, length(ppo.envs)
    experiment_progress = length(returns) / max_trials
    if ppo.decay_lr
        actor_lr, critic_lr = (ppo.lr_actor, ppo.lr_critic) .* (1 - experiment_progress)
        actor_lr, critic_lr = max(actor_lr, ppo.min_lr), max(critic_lr, ppo.min_lr)
        ppo.optim_actor[end].eta, ppo.optim_critic[end].eta = actor_lr, critic_lr
        ppo.stats[:lr_actor], ppo.stats[:lr_critic] = actor_lr, critic_lr
    end
    entropy_bonus = ppo.decay_ent_bonus ? ppo.entropy_bonus * (1 - experiment_progress) : ppo.entropy_bonus
    if ppo.actor.recurtype != MARKOV && ppo.batch_size % M != 0
        new_batch_size = clamp(round(Int, ppo.batch_size / M) * M, M, M * N)
        @warn "batch_size ($(ppo.batch_size)) is not a multiple of nsteps ($M). Changing batch_size to $new_batch_size"
        ppo.batch_size = new_batch_size
    end

    function do_full_ppo_iteration()
        𝐬, 𝐚, 𝛑, log𝛑, 𝐫, 𝐭, 𝐝 = collect_trajectories(ppo, ppo.actor_gpu, entropy_bonus, ppo.device, rng) |> ppo.device
        if eltype(𝐚) <: Integer; 𝐚 = cpu(𝐚); end
        Flux.reset!(ppo.critic_gpu)
        𝐯, 𝛅 = get_values_advantages(ppo, ppo.critic_gpu, 𝐬, 𝐫, 𝐭, 𝐝, ppo.γ, ppo.λ)

        stop_actor_training, kl = false, 0f0
        losses, actor_losses, critic_losses = [], [], []
        θ = Flux.params(ppo.actor_gpu, ppo.critic_gpu)
        while length(losses) < ppo.nepochs
            desc = stop_actor_training ? "Train Critic Epoch $(length(losses)+1)" : "Train Actor-Critic Epoch $(length(losses)+1)"
            progress = Progress(M * N; desc=desc, color=:magenta, enabled=ppo.progressmeter)
            loss, actor_loss, critic_loss = 0, 0, 0
            data_indices = cartesian_product(M, N)
            data_indices = ppo.actor.recurtype == MARKOV ? shuffle(rng, data_indices) : data_indices
            for mb_indices in partition(data_indices, ppo.batch_size)
                mb_𝐬, mb_𝐚, mb_𝐯, mb_𝛅, mb_𝛑, mb_log𝛑 = map(𝐱 -> 𝐱[:, mb_indices], (𝐬, 𝐚, 𝐯, 𝛅, 𝛑, log𝛑))
                if ppo.actor.recurtype != MARKOV 
                    mb_𝐬, mb_𝐚, mb_𝐯, mb_𝛅, mb_𝛑, mb_log𝛑 = map(𝐱 -> reshape(𝐱, :, M, length(mb_indices) ÷ M), (mb_𝐬, mb_𝐚, mb_𝐯, mb_𝛅, mb_𝛑, mb_log𝛑))
                end
                Flux.reset!(ppo.actor_gpu)
                Flux.reset!(ppo.critic_gpu)
                ∇ = gradient(θ) do
                    mb_loss, mb_actor_loss, mb_critic_loss = ppo_loss(ppo, ppo.actor_gpu, ppo.critic_gpu, mb_𝐬, mb_𝐚, mb_𝐯, mb_𝛅, mb_𝛑, mb_log𝛑, Float32(!stop_actor_training), 0.5f0, entropy_bonus)
                    actor_loss += mb_actor_loss * length(mb_indices) / (M * N)
                    critic_loss += mb_critic_loss * length(mb_indices) / (M * N)
                    loss += mb_loss * length(mb_indices) / (M * N)
                    return mb_loss
                end
                ppo.clipnorm < Inf && clip_global_norm!(∇, θ, ppo.clipnorm)
                !stop_actor_training && Flux.update!(ppo.optim_actor, Flux.params(ppo.actor_gpu), ∇)
                Flux.update!(ppo.optim_critic, Flux.params(ppo.critic_gpu), ∇)
                next!(progress; step=length(mb_indices))
            end
            finish!(progress)
            push!(losses, loss)
            push!(critic_losses, critic_loss)
            if !stop_actor_training
                push!(actor_losses, actor_loss)
                Flux.reset!(ppo.actor_gpu)
                kl = get_kl_div(ppo.actor_gpu, 𝐬, 𝐚, 𝛑, log𝛑)
                stop_actor_training = kl >= ppo.kl_target
            end
            stop_actor_training && ppo.early_stop_critic && break
        end
        Flux.reset!(ppo.actor_gpu)
        Flux.reset!(ppo.critic_gpu)
        H̄, v̄ = mean(get_entropy(ppo.actor_gpu, 𝐬)), mean(get_values(ppo.critic_gpu, 𝐬, ppo.actor.recurtype))

        Flux.loadparams!(ppo.actor, Flux.params(ppo.actor_gpu))
        Flux.loadparams!(ppo.critic, Flux.params(ppo.critic_gpu))

        ppo.stats[:ℓ] = losses[end]
        ppo.stats[:actor_loss] = actor_losses[end]
        ppo.stats[:critic_loss] = critic_losses[end]
        ppo.stats[:H̄] = H̄
        ppo.stats[:iteration_kl] = kl
        ppo.stats[:v̄] = v̄
        ppo.stats[:iteration_actor_epochs] = length(actor_losses)
        ppo.stats[:iteration_critic_epochs] = length(critic_losses)
        ppo.stats[:iterations] += 1
        ppo.stats[:ent_bonus] = entropy_bonus
        ppo.stats[:iteration_R̄] = mean(sum(𝐫, dims=2))
        ppo.stats[:iteration_r̄] = ppo.stats[:iteration_R̄] / M
        if ppo.actor isa PPOActorContinuous
            ppo.stats[:logstd] = string(ppo.actor.logstd)
        end
    end

    for iter_num in 1:ppo.iters_per_postepisode
        do_full_ppo_iteration()
    end

    nothing
end


function collect_trajectories(ppo::PPOLearner, actor, ent_coeff, device, rng)
    state_dim = size(state_space(ppo.envs), 1)
    isdiscrete = action_space(ppo.envs) isa IntegerSpace
    if isdiscrete
        nactions = length(action_space(ppo.envs))
    else
        action_dim = size(action_space(ppo.envs), 1)
    end
    M, N = ppo.nsteps, length(ppo.envs)

    if state_dim > 50
        𝐬 = zeros(Float32, state_dim, M, N) |> device
    else
        𝐬 = zeros(Float32, state_dim, M, N)
    end
    if isdiscrete
        𝐚 = zeros(Int, 1, M, N)
        𝛑 = zeros(Float32, nactions, M, N)
        log𝛑 = zeros(Float32, nactions, M, N)
    else
        𝐚 = zeros(Float32, action_dim, M, N)
        𝛑 = zeros(Float32, 1, M, N)
        log𝛑 = zeros(Float32, 1, M, N)
    end
    𝐫 = zeros(Float32, 1, M, N)
    𝐭 = zeros(Float32, 1, M, N)
    𝐝 = zeros(Float32, 1, M, N)

    progress = Progress(M; color=:white, desc="(Iter $(ppo.stats[:iterations] + 1)) Collecting trajectories", enabled=ppo.progressmeter)

    Flux.reset!(actor)
    for t in 1:M
        reset!(ppo.envs, false; rng=rng)
        𝐬ₜ = state(ppo.envs) |> tof32
        if state_dim > 50
            𝐬[:, t, :] .= device(𝐬ₜ)
        else
            𝐬[:, t, :] .= 𝐬ₜ
        end
        if isdiscrete
            @assert actor isa PPOActorDiscrete
            if ppo.actor.recurtype ∈ (MARKOV, RECURRENT)
                𝛑ₜ, log𝛑ₜ = get_probs_logprobs(actor, device(𝐬ₜ)) |> cpu
            elseif ppo.actor.recurtype == TRANSFORMER
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
            𝛑[:, t, :] = 𝛑ₜ
            log𝛑[:, t, :] = log𝛑ₜ
            ents = get_entropy(actor, 𝛑ₜ, log𝛑ₜ)
        else
            @assert actor isa PPOActorContinuous
            if ppo.actor.recurtype ∈ (MARKOV, RECURRENT)
                𝐚ₜ, log𝛑ₜ, log𝛔ₜ = sample_action_logprobs(actor, rng, device(𝐬ₜ); return_logstd=true) |> cpu
            else
                𝐚ₜ, log𝛑ₜ, log𝛔ₜ = sample_action_logprobs(actor, rng, device(𝐬[:, 1:t, :]); return_logstd=true) |> cpu
                𝐚ₜ, log𝛑ₜ = 𝐚ₜ[:, end, :], log𝛑ₜ[:, end, :]
            end
            𝐚[:, t, :] = 𝐚ₜ
            log𝛑[:, t, :] = log𝛑ₜ
            𝛑 = exp.(log𝛑)
            ents = get_gaussian_entropy(log𝛔ₜ)
        end

        if isdiscrete
            _𝐚ₜ = 𝐚ₜ[1, :]
        else
            Tₐ = ppo.envs |> action_space |> eltype |> eltype
            _𝐚ₜ = convert(Matrix{Tₐ}, 𝐚ₜ) |> eachcol .|> copy    # eachcol makes it a vector of vectors
            _𝐚ₜ = convert(Matrix{Tₐ}, 𝐚ₜ)
        end
        step!(ppo.envs, _𝐚ₜ; rng=rng)
        𝐫ₜ = reward(ppo.envs)' |> tof32
        if ppo.entropy_method == :maximized && ent_coeff > 0
            @assert size(ents) == size(𝐫ₜ)  "size mismatch: $(size(ents)) != $(size(𝐫ₜ))"
            𝐫ₜ += ent_coeff * ents
        end
        𝐫[:, t, :] = 𝐫ₜ
        𝐭ₜ = in_absorbing_state(ppo.envs)' |> tof32
        𝐭[:, t, :] = 𝐭ₜ
        𝐝ₜ = (in_absorbing_state(ppo.envs) .|| truncated(ppo.envs))' |> tof32
        𝐝[:, t, :] = 𝐝ₜ

        next!(progress)
    end
    finish!(progress)
    return 𝐬, 𝐚, 𝛑, log𝛑, 𝐫, 𝐭, 𝐝
end


function get_values_advantages(ppo::PPOLearner, critic, 𝐬, 𝐫, 𝐭, 𝐝, γ, λ)
    𝐬ₜ′ = unsqueeze(state(ppo.envs), dims=2) |> tof32
    𝐬ₜ′ = convert(typeof(𝐬), 𝐬ₜ′)
    _𝐯 = get_values(critic, hcat(𝐬, 𝐬ₜ′), ppo.actor.recurtype)
    𝐯, 𝐯′ = _𝐯[:, 1:end-1, :], _𝐯[:, 2:end, :]
    𝛅 = 𝐫 + γ * (1 .- 𝐭) .* 𝐯′ - 𝐯
    should_ignore_error = 𝐝 - 𝐭  # if a trajectory was truncated, set δ = 0. This state should not contribute to the loss.
    should_ignore_error[1, end, :] .= 0 # since we handled the last state of each trajectory properly
    𝛅 .*= (1f0 .- should_ignore_error)
    𝐀ₜ′ = 0
    for t in reverse(1:size(𝛅, 2))
        𝛅[:, t:t, :] .+= (1f0 .- 𝐝[:, t:t, :]) .* γ .* λ .* 𝐀ₜ′
        𝐀ₜ′ = 𝛅[:, t:t, :]
    end
    return 𝐯, 𝛅
end


function ppo_loss(ppo::PPOLearner, actor, critic, 𝐬, 𝐚, 𝐯, 𝛅, old𝛑, oldlog𝛑, actor_coeff, critic_coeff, ent_coeff)
    critic_loss = critic_coeff > 0 ? get_critic_loss(critic, 𝐬, 𝐯, 𝛅, ppo.actor.recurtype) : 0f0
    𝛅 = !ppo.normalize_advantages ? 𝛅 : Zygote.@ignore (𝛅 .- mean(𝛅)) ./ (std(𝛅) + 1e-8)
    actor_loss, entropy = actor_coeff > 0 ? get_loss_and_entropy(actor, 𝐬, 𝐚, 𝛅, old𝛑, oldlog𝛑, ppo.ϵ, ppo.ppo) : (0f0, 0f0)
    total_loss = actor_loss + critic_coeff * critic_loss
    if ppo.entropy_method == :regularized && ent_coeff > 0
        total_loss -= ent_coeff * entropy
    end
    return total_loss, actor_loss, critic_loss
end