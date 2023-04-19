using MDPs
import MDPs: preexperiment, postepisode, poststep
using UnPack
using Random
using Flux
using Flux.Zygote

export PPOTransformerLearner

"""
Right now, this code assumes that horizon <= nsteps, because before each rollout in the training, the environments are reset, which implies that if horizon was too long, the agent would not get to see horizon lenngth trajectories as they are played only up to nsteps.
"""
Base.@kwdef mutable struct PPOTransformerLearner <: AbstractHook
    envs::Vector{AbstractMDP}   # A vector of differently seeded environments.
    actor::TransformerActor
    critic                      # some transformer model
    γ::Float32 = 0.99           # discount factor. Used to calulate TD(λ) advantages.
    nsteps::Int = 100           # numer of steps per iteration. So that total data per iteration = nenvs * nsteps. With longer nsteps, TD(λ) returns are computed better.
    nepochs::Int = 10           # number of epochs per iteration.
    minibatch_size::Int = 32    # number of trajectories per minibatch in transformer training.
    entropy_bonus::Float32 = 0.01f0 # coefficient of the entropy term in the overall PPO loss, to encourage exploration.
    center_advantages::Bool = true # whether to center advantages to have zero mean.
    scale_advantages::Bool = true  # whether to scale advantages to have unit variance.
    clipnorm::Float32 = Inf     # clip gradients by norm
    adam_weight_decay::Float32 = 0f0      # adam weight decay
    adam_epsilon::Float32 = 1f-7    # adam epsilon
    lr::Float32 = 0.0003        # adam learning rate
    λ = 0.95f0                  # Used to calulate TD(λ) advantages using Generalized Advantage Estimation (GAE) method.
    ϵ = 0.2f0                   # epsilon used in PPO clip objective
    kl_target = 0.01            # In each iteration, early stop training if KL divergence from old policy exceeds this value.
    critic_early_stop::Bool = false  # whether to early stop training critic if KL divergence from old policy exceeds kl_target or train for nepochs.
    ppo = true                  # whether to use PPO clip objective. If false, standard advantange actor-critic (A2C) objective will be used.
    device = cpu                # `cpu` or `gpu`
    progressmeter::Bool = false # Whether to show data and gradient updates progress using a progressmeter

    # data structures:
    optim_actor = make_adam_optim(lr, (0.9, 0.999), adam_epsilon, clipnorm, 0)
    optim_critic = make_adam_optim(lr, (0.9, 0.999), adam_epsilon, clipnorm, adam_weight_decay)  # regularize critic with weight decay (l2 norm) but don't regularize actor
    actor_gpu = device(deepcopy(actor))
    critic_gpu = device(deepcopy(critic))

    stats = Dict{Symbol, Any}()
end


function postepisode(ppo::PPOTransformerLearner; returns, steps, rng, kwargs...)
    @unpack envs, γ, nsteps, nepochs, minibatch_size, entropy_bonus, λ, ϵ, kl_target, device, progressmeter = ppo

    m = size(state_space(envs[1]), 1)
    n = size(action_space(envs[1]), 1)
    e = n+1+1+m
    N = length(envs)
    M = nsteps
    episodes = length(returns)

    function collect_trajectories(actor, critic)    
        Threads.@threads for i in 1:N; reset!(envs[i], rng=rng); end
        𝐚ₜ_prev = zeros(Float32, n, N) # previous action (onehot) of each env
        𝐫ₜ_prev = zeros(Float32, 1, N) # previous reward of each env
        𝐝ₜ_prev = ones(Float32, 1, N) # previous done of each env
        𝐬ₜ = mapfoldl(state, hcat, envs) |> tof32 # state of each env (m, N)
    
        𝐞 = zeros(Float32, e, 0, N)         # cat of a_prev, r_prev, d_prev, s.   (e, 0, N) to begin with
        𝐚 = zeros(Int, 1, 0, N)             # action at each step.          (1, 0, N) to begin with
        𝐫 = zeros(Float32, 1, 0, N)         # reward after each step.       (1, 0, N) to begin with
        𝐭 = zeros(Float32, 1, 0, N)         # terminal after each step.     (1, 0, N) to begin with. "terminal" means the env transitioned to a terminal state.
        𝐝 = zeros(Float32, 1, 0, N)         # done after each step.         (1, 0, N) to begin with. "done" means end of episode, which can be due to either env being truncated or transitioned to terminal state.
        𝐯′ = zeros(Float32, 1, 0, N)        # value of next state.          (1, 0, N) to begin with
        
    
        progress = Progress(M; color=:white, desc="Collecting trajectories", enabled=progressmeter)

        for t in 1:M
            𝐞ₜ = vcat(𝐚ₜ_prev, 𝐫ₜ_prev, 𝐝ₜ_prev, 𝐬ₜ)                        # (e, N)
            @assert size(𝐞ₜ) == (e, N)
            𝐞 = cat(𝐞, reshape(𝐞ₜ, :, 1, N); dims=2)                        # (e, t, N)
            @assert size(𝐞) == (e, t, N)

            𝛑ₜ, log𝛑ₜ = cpu(get_probs_logprobs(actor, device(𝐞)))                        # (n, t, N)
            @assert size(𝛑ₜ) == (n, t, N)
            @assert size(log𝛑ₜ) == (n, t, N)
            𝐚ₜ = zeros(Int, 1, N)
            Threads.@threads for i in 1:N
                aᵢₜ = sample(rng, 1:n, ProbabilityWeights(𝛑ₜ[:, end, i]))
                step!(envs[i], aᵢₜ; rng=rng)
                𝐚ₜ[1, i] = aᵢₜ
            end
            𝐚ₜ_onehot = @views Flux.OneHotArrays.onehotbatch(𝐚ₜ[1, :], 1:n) |> tof32 # (n, N)
            @assert size(𝐚ₜ_onehot) == (n, N)

            𝐚 = cat(𝐚, reshape(𝐚ₜ, 1, 1, N); dims=2)                        # (1, t, N)
            @assert size(𝐚) == (1, t, N)
            
            𝐫ₜ = mapfoldl(reward, hcat, envs) |> tof32                      # (1, N)
            @assert size(𝐫ₜ) == (1, N)
            𝐫 = cat(𝐫, reshape(𝐫ₜ, 1, 1, N); dims=2)                        # (1, t, N)
            @assert size(𝐫) == (1, t, N)

            𝐭ₜ = mapfoldl(in_absorbing_state, hcat, envs) |> tof32          # (1, N)
            @assert size(𝐭ₜ) == (1, N)
            𝐭 = cat(𝐭, reshape(𝐭ₜ, 1, 1, N); dims=2)                        # (1, t, N)
            @assert size(𝐭) == (1, t, N)

            𝐝ₜ = mapfoldl(env -> in_absorbing_state(env) || truncated(env), hcat, envs) |> tof32 # (1, N)
            @assert size(𝐝ₜ) == (1, N)
            𝐝 = cat(𝐝, reshape(𝐝ₜ, 1, 1, N); dims=2)                        # (1, t, N)
            @assert size(𝐝) == (1, t, N)

            𝐬ₜ′= mapfoldl(state, hcat, envs) |> tof32                       # (m, N)
            @assert size(𝐬ₜ′) == (m, N)
            𝐞ₜ′ = vcat(𝐚ₜ_onehot, 𝐫ₜ, zeros(Float32, 1, N), 𝐬ₜ′)            # (e, N)
            @assert size(𝐞ₜ′) == (e, N)
            𝐞′ = cat(𝐞, reshape(𝐞ₜ′, e, 1, N); dims=2)                      # (e, t + 1, N)
            @assert size(𝐞′) == (e, t+1, N)
            𝐯′ₜ = cpu(critic(device(𝐞′)))[:, end, :]                                     # (1, N)
            @assert size(𝐯′ₜ) == (1, N)
            𝐯′ = cat(𝐯′, reshape(𝐯′ₜ, 1, 1, N); dims=2)                     # (1, t, N)
            @assert size(𝐯′) == (1, t, N)


            # --------------------- for next step --------------------
            𝐚ₜ_prev = 𝐚ₜ_onehot
            𝐫ₜ_prev = 𝐫ₜ
            𝐝ₜ_prev = 𝐝ₜ
            𝐬ₜ = 𝐬ₜ′
            for i in 1:N
                if 𝐝ₜ[1, i] == 1
                    reset!(envs[i], rng=rng)
                    𝐚ₜ_prev[:, i] .= 0
                    𝐫ₜ_prev[1, i] = 0
                    𝐝ₜ_prev[1, i] = 1
                    𝐬ₜ[:, i] .= state(envs[i]) |> tof32
                end
            end
            # ---------------- prepare for next step -------------------
            
            next!(progress)
        end
        finish!(progress)
        return 𝐞, 𝐚, 𝐫, 𝐭, 𝐝, 𝐯′
    end

    function get_advantages_value_pi_logpi(actor, critic, 𝐞, 𝐚, 𝐫, 𝐭, 𝐝, 𝐯′)
        𝛑, log𝛑 = get_probs_logprobs(actor, 𝐞)                              # (n, M, N)
        𝐯 = critic(𝐞)                                                       # (1, M, N)
        𝛅 = 𝐫 + γ * (1 .- 𝐭) .* 𝐯′ - 𝐯                                      # (1, M, N)
        # update advantages using GAE
        𝐀ₜ′ = 0
        for t in reverse(1:M)
            𝛅[:, t:t, :] .+= (1f0 .- 𝐝[:, t:t, :]) .* γ .* λ .* 𝐀ₜ′
            𝐀ₜ′ = 𝛅[:, t:t, :]
        end
        return 𝛅, 𝐯, 𝛑, log𝛑
    end

    """
    `𝐞` is the evidence tensor of shape (e, seq_len, batch_size)
    `𝐚` is the Int action tensor of shape (1, seq_len, batch_size)
    `𝛅` is the advantage tensor of shape (1, seq_len, batch_size)
    `old𝛑` is the old policy tensor of shape (n, seq_len, batch_size)
    `𝐯` is the value tensor of shape (1, seq_len, batch_size)
    """
    function ppo_loss(actor, critic, 𝐞, 𝐚, 𝛅, old𝛑, 𝐯)
        # ---- critic loss ----
        critic_loss = Flux.mse(critic(𝐞), 𝐯 + 𝛅)
        # ---- actor loss ----
        _, seq_len, batch_size = size(𝐚)
        𝐚 = Zygote.@ignore [CartesianIndex(𝐚[1, t, i], t, i) for t in 1:seq_len, i in 1:batch_size] # (seq_len, batch_size)
        𝐚 = reshape(𝐚, 1, seq_len, batch_size)  # (1, seq_len, batch_size)
        𝛑, log𝛑 = get_probs_logprobs(actor, 𝐞) # (n, seq_len, batch_size)
        𝛅 = !ppo.center_advantages ? 𝛅 : Zygote.@ignore 𝛅 .- mean(𝛅) # (1, seq_len, batch_size)
        𝛅 = !ppo.scale_advantages ? 𝛅 : Zygote.@ignore 𝛅 ./ (std(𝛅) + 1e-8) # (1, seq_len, batch_size)
        if ppo.ppo
            𝑟 =  𝛑[𝐚] ./ old𝛑[𝐚] # (1, seq_len, batch_size)
            actor_loss = -min.(𝑟 .* 𝛅, clamp.(𝑟, 1-ϵ, 1+ϵ) .* 𝛅) |> mean
        else
            actor_loss = -𝛅 .* log𝛑[𝐚] |> mean
        end
        # ---- entropy bonus ----
        entropy = -sum(𝛑 .* log𝛑; dims=1) |> mean
        # ---- total loss ----
        return actor_loss + critic_loss - entropy_bonus * entropy, actor_loss, critic_loss
    end

    """
    `𝐞` is the evidence tensor of shape (e, seq_len, batch_size)
    `𝐚` is the Int action tensor of shape (1, seq_len, batch_size)
    `𝛅` is the advantage tensor of shape (1, seq_len, batch_size)
    `old𝛑` is the old policy tensor of shape (n, seq_len, batch_size)
    `𝐯` is the value tensor of shape (1, seq_len, batch_size)
    """
    function ppo_loss_actor_only(actor, 𝐞, 𝐚, 𝛅, old𝛑, 𝐯)
        # ---- actor loss ----
        _, seq_len, batch_size = size(𝐚)
        𝐚 = Zygote.@ignore [CartesianIndex(𝐚[1, t, i], t, i) for t in 1:seq_len, i in 1:batch_size] # (seq_len, batch_size)
        𝐚 = reshape(𝐚, 1, seq_len, batch_size)  # (1, seq_len, batch_size)
        𝛑, log𝛑 = get_probs_logprobs(actor, 𝐞) # (n, seq_len, batch_size)
        𝛅 = !ppo.center_advantages ? 𝛅 : Zygote.@ignore 𝛅 .- mean(𝛅) # (1, seq_len, batch_size)
        𝛅 = !ppo.scale_advantages ? 𝛅 : Zygote.@ignore 𝛅 ./ (std(𝛅) + 1e-8) # (1, seq_len, batch_size)
        if ppo.ppo
            𝑟 =  𝛑[𝐚] ./ old𝛑[𝐚] # (1, seq_len, batch_size)
            actor_loss = -min.(𝑟 .* 𝛅, clamp.(𝑟, 1-ϵ, 1+ϵ) .* 𝛅) |> mean
        else
            actor_loss = -𝛅 .* log𝛑[𝐚] |> mean
        end
        # ---- entropy bonus ----
        entropy = -sum(𝛑 .* log𝛑; dims=1) |> mean
        return actor_loss - entropy_bonus * entropy, actor_loss
    end

    """
    `𝐞` is the evidence tensor of shape (e, seq_len, batch_size)
    `𝐚` is the Int action tensor of shape (1, seq_len, batch_size)
    `𝛅` is the advantage tensor of shape (1, seq_len, batch_size)
    `old𝛑` is the old policy tensor of shape (n, seq_len, batch_size)
    `𝐯` is the value tensor of shape (1, seq_len, batch_size)
    """
    function ppo_loss_critic_only(critic, 𝐞, 𝐚, 𝛅, old𝛑, 𝐯)
        # ---- critic loss ----
        critic_loss = Flux.mse(critic(𝐞), 𝐯 + 𝛅)
        return critic_loss, critic_loss
    end

    """
    `𝐞` is the evidence tensor of shape (e, M, N)
    `𝐚` is the Int action tensor of shape (1, M, N)
    `𝛅` is the advantage tensor of shape (1, M, N)
    `old𝛑` is the old policy tensor of shape (n, M, N)
    `𝐯` is the value tensor of shape (1, M, N)
    """
    function update_actor_critic_one_epoch!(actor, critic, 𝐞, 𝐚, 𝛅, old𝛑, 𝐯)
        loss, actor_loss, critic_loss, θ_actor, θ_critic, θ = 0f0, 0f0, 0f0, Flux.params(actor), Flux.params(critic), Flux.params(actor, critic)
        nsgdsteps = ceil(Int, N / ppo.minibatch_size)
        _N = ceil(Int, N / nsgdsteps) # num envs per minibatch
        progress = Progress(N; desc="Performing actor critic updates", color=:magenta, enabled=progressmeter)
        for env_indices in splitequal(N, _N)
            _𝐞, _𝐚, _𝛅, _old𝛑, _𝐯 = @views map(𝐱 -> 𝐱[:, :, env_indices], (𝐞, 𝐚, 𝛅, old𝛑, 𝐯)) # (e, M, _N)
            _loss, _∇ =  withgradient(θ) do
                _l, _al, _cl = ppo_loss(actor, critic, _𝐞, _𝐚, _𝛅, _old𝛑, _𝐯)
                actor_loss += _al * length(env_indices) / N
                critic_loss += _cl * length(env_indices) / N
                loss += _l * length(env_indices) / N
                return _l
            end
            Flux.update!(ppo.optim_actor, θ_actor, _∇)
            Flux.update!(ppo.optim_critic, θ_critic, _∇)
            next!(progress; step=length(env_indices))
        end
        finish!(progress)
        return loss, actor_loss, critic_loss
    end

    """
    `𝐞` is the evidence tensor of shape (e, M, N)
    `𝐚` is the Int action tensor of shape (1, M, N)
    `𝛅` is the advantage tensor of shape (1, M, N)
    `old𝛑` is the old policy tensor of shape (n, M, N)
    `𝐯` is the value tensor of shape (1, M, N)
    """
    function update_actor_only_one_epoch!(actor, 𝐞, 𝐚, 𝛅, old𝛑, 𝐯)
        loss, actor_loss, θ = 0f0, 0f0, Flux.params(actor)
        nsgdsteps = ceil(Int, N / ppo.minibatch_size)
        _N = ceil(Int, N / nsgdsteps) # num envs per minibatch
        progress = Progress(N; desc="Performing actor updates", color=:blue, enabled=progressmeter)
        for env_indices in splitequal(N, _N)
            _𝐞, _𝐚, _𝛅, _old𝛑, _𝐯 = @views map(𝐱 -> 𝐱[:, :, env_indices], (𝐞, 𝐚, 𝛅, old𝛑, 𝐯)) # (e, M, _N)
            _loss, _∇ =  withgradient(θ) do
                _l, _al = ppo_loss_actor_only(actor, _𝐞, _𝐚, _𝛅, _old𝛑, _𝐯)
                actor_loss += _al * length(env_indices) / N
                loss += _l * length(env_indices) / N
                return _l
            end
            Flux.update!(ppo.optim_actor, θ, _∇)
            next!(progress; step=length(env_indices))
        end
        finish!(progress)
        return loss, actor_loss
    end

    """
    `𝐞` is the evidence tensor of shape (e, M, N)
    `𝐚` is the Int action tensor of shape (1, M, N)
    `𝛅` is the advantage tensor of shape (1, M, N)
    `old𝛑` is the old policy tensor of shape (n, M, N)
    `𝐯` is the value tensor of shape (1, M, N)
    """
    function update_critic_only_one_epoch!(critic, 𝐞, 𝐚, 𝛅, old𝛑, 𝐯)
        loss, critic_loss, θ = 0f0, 0f0, Flux.params(critic)
        nsgdsteps = ceil(Int, N / ppo.minibatch_size)
        _N = ceil(Int, N / nsgdsteps) # num envs per minibatch
        progress = Progress(N; desc="Performing critic updates", color=:red, enabled=progressmeter)
        for env_indices in splitequal(N, _N)
            _𝐞, _𝐚, _𝛅, _old𝛑, _𝐯 = @views map(𝐱 -> 𝐱[:, :, env_indices], (𝐞, 𝐚, 𝛅, old𝛑, 𝐯)) # (e, M, _N)
            _loss, _∇ =  withgradient(θ) do
                _l, _cl = ppo_loss_critic_only(critic, _𝐞, _𝐚, _𝛅, _old𝛑, _𝐯)
                critic_loss += _cl * length(env_indices) / N
                loss += _l * length(env_indices) / N
                return _l
            end
            Flux.update!(ppo.optim_critic, θ, _∇)
            next!(progress; step=length(env_indices))
        end
        finish!(progress)
        return critic_loss
    end

    

    """
    `𝐞` is the evidence tensor of shape (e, M, N)
    `old𝛑` is the old policy tensor of shape (n, M, N)

    returns the KL divergence, the mean entropy, and the mean value
    """
    function calculate_stats(actor, critic, 𝐞, old𝛑, oldlog𝛑)
        𝛑, log𝛑 = get_probs_logprobs(actor, 𝐞) # (n, M, N)
        H̄ = -sum(𝛑 .* log𝛑; dims=1) |> mean
        kl = sum(old𝛑 .* (oldlog𝛑 .- log𝛑); dims=1) |> mean
        v̄ = critic(𝐞) |> mean
        v̄₁ = @views sum(critic(𝐞)[1, 1, :]) / N
        return kl, H̄, v̄, v̄₁
    end

    function update_actor_critic_with_early_stopping!(actor, critic, epochs, 𝐞, 𝐚, 𝛅, old𝛑, oldlog𝛑, 𝐯)
        ℓ, actor_loss, critic_loss, kl, H̄, v̄, v̄₁ = 0, 0, 0, 0, 0, 0, 0
        num_epochs = 0
        for epoch in 1:epochs
            ℓ, actor_loss, critic_loss  = update_actor_critic_one_epoch!(actor, critic, 𝐞, 𝐚, 𝛅, old𝛑, 𝐯)
            num_epochs += 1
            kl, H̄, v̄, v̄₁ = calculate_stats(actor, critic, 𝐞, old𝛑, oldlog𝛑)
            kl >= kl_target && break
        end
        return ℓ, actor_loss, critic_loss, kl, H̄, v̄, v̄₁, num_epochs
    end

    function update_actor_with_early_stopping_and_critic_full!(actor, critic, epochs, 𝐞, 𝐚, 𝛅, old𝛑, oldlog𝛑, 𝐯)
        ℓ, actor_loss_with_ent_bonus, actor_loss, critic_loss, kl, H̄, v̄, v̄₁ = 0, 0, 0, 0, 0, 0, 0, 0
        num_epochs = 0
        for epoch in 1:epochs
            ℓ, actor_loss, critic_loss  = update_actor_critic_one_epoch!(actor, critic, 𝐞, 𝐚, 𝛅, old𝛑, 𝐯)
            actor_loss_with_ent_bonus = ℓ - critic_loss
            num_epochs += 1
            kl, H̄, v̄, v̄₁ = calculate_stats(actor, critic, 𝐞, old𝛑, oldlog𝛑)
            kl >= kl_target && break
        end
        for epoch in (num_epochs + 1):epochs # continue training critic
            critic_loss = update_critic_only_one_epoch!(critic, 𝐞, 𝐚, 𝛅, old𝛑, 𝐯)
        end
        ℓ = actor_loss_with_ent_bonus + critic_loss
        kl, H̄, v̄, v̄₁ = calculate_stats(actor, critic, 𝐞, old𝛑, oldlog𝛑)
        return ℓ, actor_loss, critic_loss, kl, H̄, v̄, v̄₁, num_epochs
    end

    data_𝐞, data_𝐚, data_𝐫, data_𝐭, data_𝐝, data_𝐯′ = collect_trajectories(ppo.actor_gpu, ppo.critic_gpu)
    data_𝐞, data_𝐫, data_𝐭, data_𝐝, data_𝐯′ = device.((data_𝐞, data_𝐫, data_𝐭, data_𝐝, data_𝐯′))
    data_𝛅, data_𝐯, data_𝛑, data_log𝛑 = get_advantages_value_pi_logpi(ppo.actor_gpu, ppo.critic_gpu, data_𝐞, data_𝐚, data_𝐫, data_𝐭, data_𝐝, data_𝐯′)

    if ppo.critic_early_stop
        ℓ, actor_loss, critic_loss, kl, H̄, v̄, v̄₁, num_epochs = update_actor_critic_with_early_stopping!(ppo.actor_gpu, ppo.critic_gpu, ppo.nepochs, data_𝐞, data_𝐚, data_𝛅, data_𝛑, data_log𝛑, data_𝐯)
    else
        ℓ, actor_loss, critic_loss, kl, H̄, v̄, v̄₁, num_epochs = update_actor_with_early_stopping_and_critic_full!(ppo.actor_gpu, ppo.critic_gpu, ppo.nepochs, data_𝐞, data_𝐚, data_𝛅, data_𝛑, data_log𝛑, data_𝐯)
    end
    
    Flux.loadparams!(ppo.actor, Flux.params(ppo.actor_gpu))
    Flux.loadparams!(ppo.critic, Flux.params(ppo.critic_gpu))

    ppo.stats[:ℓ] = ℓ
    ppo.stats[:actor_loss] = actor_loss
    ppo.stats[:critic_loss] = critic_loss
    ppo.stats[:H̄] = H̄
    ppo.stats[:kl] = kl
    ppo.stats[:v̄] = v̄
    ppo.stats[:v̄₁] = v̄₁
    ppo.stats[:num_epochs] = num_epochs

    @debug "learning stats" steps episodes stats...
    nothing
end