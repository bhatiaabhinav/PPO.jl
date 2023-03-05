using MDPs
import MDPs: preexperiment, postepisode, poststep
using UnPack
using Random
using Flux
using Flux.Zygote
using Flux: Optimiser, ClipNorm

export PPOLearner

Base.@kwdef mutable struct PPOLearner <: AbstractHook
    envs::Vector{AbstractMDP}   # A vector of differently seeded environments.
    actor::Union{Actor, RecurrentActor}       # `Actor` maps states to action probabilities, and the input size of dimensionality of state space. `RecurrentActor` maps catenation of [latest action (onehot), latest reward, current state] to action probabilities, and therefore the input size is: number of actions + 1 + dimensionality of states
    critic                      # Any Flux model mapping 𝑆 -> ℝ. If the actor is a `RecurrentActor`, the critic is also expected to be a recurrent model mapping ℝ^|𝐴| × ℝ × 𝑆 -> ℝ. i.e., like for the actor, the inputs are a catenation of [latest action (onehot), latest reward, current state].
    γ::Float32 = 0.99           # discount factor. Used to calulate TD(λ) advantages.
    nsteps::Int = 100           # numer of steps per iteration. So that total data per iteration = nenvs * nsteps. With longer nsteps, TD(λ) returns are computed better.
    nepochs::Int = 10           # number of epochs per iteration.
    batch_size::Int = 256       # will be clipped to be between nenvs and nenvs * nsteps. Accordingly, nsteps per batch (which is truncated backprop horizon in case of RNNs) = batch_size / nenvs.
    entropy_bonus::Float32 = 0.01f0 # coefficient of the entropy term in the overall PPO loss, to encourage exploration.
    clipnorm::Float32 = Inf     # clip gradients by norm
    adam_weight_decay::Float32 = 0f0      # adam weight decay
    adam_epsilon::Float32 = 1f-7    # adam epsilon
    lr::Float32 = 0.0003        # adam learning rate
    λ = 0.95f0                  # Used to calulate TD(λ) advantages using Generalized Advantage Estimation (GAE) method.
    ϵ = 0.2f0                   # epsilon used in PPO clip objective
    kl_target = 0.01            # In each iteration, early stop training if KL divergence from old policy exceeds this value.
    ppo = true                  # whether to use PPO clip objective. If false, standard advantange actor-critic (A2C) objective will be used.
    device = cpu                # `cpu` or `gpu`

    # data structures:
    optim = make_adam_optim(lr, (0.9, 0.999), adam_epsilon, clipnorm, adam_weight_decay)
    actor_gpu = device(deepcopy(actor))
    critic_gpu = device(deepcopy(critic))
    𝐡ₜ = nothing                # hidden states of the RNNs between training iterations
    𝐬ₜ = nothing                # states of envs between training iterations
    𝐯ₜ = nothing                # values of the states of the envs between training iterations
    stats = Dict{Symbol, Any}()
end


function postepisode(ppo::PPOLearner; returns, steps, rng, kwargs...)
    @unpack envs, γ, nsteps, nepochs, batch_size, entropy_bonus, λ, ϵ, kl_target, device = ppo

    isrecurrent = ppo.actor isa RecurrentActor
    m = size(state_space(envs[1]), 1)
    n = size(action_space(envs[1]), 1)
    N = length(envs)
    M = nsteps
    episodes = length(returns)

    if isnothing(ppo.𝐬ₜ)
        Threads.@threads for i in 1:N; reset!(envs[i], rng=rng); end
        𝐬ₜ = mapfoldl(state, hcat, envs) |> tof32
        if isrecurrent
            𝐬ₜ = vcat(zeros(Float32, n, N), zeros(Float32, 1, N), 𝐬ₜ)
        end
        reset_rnn_state!.((ppo.actor, ppo.critic))
        init_𝐬ₜ = 𝐬ₜ
        init_𝐡ₜ = get_rnn_state.((ppo.actor, ppo.critic))
        init_𝐯ₜ = ppo.critic(𝐬ₜ)
    else
        init_𝐬ₜ = ppo.𝐬ₜ
        init_𝐡ₜ = ppo.𝐡ₜ
        init_𝐯ₜ = ppo.𝐯ₜ
    end

    function collect_trajectories(actor, critic)    
        𝐬ₜ = init_𝐬ₜ # start where we left off
        isrecurrent && set_rnn_state!.((actor, critic), init_𝐡ₜ)
        𝐯ₜ = init_𝐯ₜ
        data = map(1:M) do t
            𝛑ₜ, log𝛑ₜ = get_probs_logprobs(actor, 𝐬ₜ)   # Forward actor
            𝐚ₜ = fill(CartesianIndex(0, 0), 1, N)
            Threads.@threads for i in 1:N
                aᵢₜ = sample(rng, 1:n, ProbabilityWeights(𝛑ₜ[:, i]))
                step!(envs[i], aᵢₜ; rng=rng)
                𝐚ₜ[1, i] = CartesianIndex(aᵢₜ, i)
            end
            𝐫ₜ = mapfoldl(reward, hcat, envs) |> tof32
            𝐬ₜ′= mapfoldl(state, hcat, envs) |> tof32
            𝐝ₜ = mapfoldl(in_absorbing_state, hcat, envs) |> tof32
            𝐭ₜ = mapfoldl(truncated, hcat, envs) |> tof32
            if isrecurrent
                𝐚ₜ_onehot = zeros(Float32, n, N)
                𝐚ₜ_onehot[𝐚ₜ] .= 1
                𝐬ₜ′ = vcat(𝐚ₜ_onehot, 𝐫ₜ, 𝐬ₜ′)
                𝐡ₜ_backup = get_rnn_state(critic)  # create a backup
            end
            𝐯ₜ′ = critic(𝐬ₜ′)
            𝛅ₜ = 𝐫ₜ + γ * (1f0 .- 𝐝ₜ) .* 𝐯ₜ′ - 𝐯ₜ
            dataₜ = (𝐬ₜ, 𝐚ₜ, 𝐫ₜ, 𝐝ₜ, 𝐭ₜ, 𝐬ₜ′, 𝛅ₜ, 𝛑ₜ, log𝛑ₜ, 𝐯ₜ)
            # ---------------- prepare for next step -------------------
            # set up states:
            𝐬ₜ = copy(𝐬ₜ′)
            any_reset = false
            for i in 1:N
                if 𝐝ₜ[1, i] + 𝐭ₜ[1, i] > 0
                    reset!(envs[i]; rng=rng);
                    𝐬ₜ[:, i] .= 0f0
                    𝐬ₜ[end-m+1:end, i] .= tof32(state(envs[i]))
                    any_reset = true
                end
            end
            # setup rnn states:
            if isrecurrent && any_reset
                set_rnn_state!(critic, 𝐡ₜ_backup)
                reset_idxs::BitVector = ((𝐝ₜ + 𝐭ₜ) .> 0)[1, :]
                reset_rnn_state!.((actor, critic), (reset_idxs, reset_idxs));
            end
            𝐯ₜ = any_reset ? critic(𝐬ₜ) : 𝐯ₜ′
            # ---------------------------------------------------------
            return dataₜ
        end
        ppo.𝐬ₜ = 𝐬ₜ
        ppo.𝐯ₜ = 𝐯ₜ
        return data
    end
        
    # update advantages using GAE
    function update_advantates!(data)
        𝐀ₜ′ = 0
        for dataₜ in reverse(data)
            (𝐬ₜ, 𝐚ₜ, 𝐫ₜ, 𝐝ₜ, 𝐭ₜ, 𝐬ₜ′, 𝛅ₜ, 𝛑ₜ, log𝛑ₜ, 𝐯ₜ) = dataₜ
            𝛅ₜ .+= γ * λ * (1f0 .- 𝐝ₜ) .* 𝐀ₜ′
            𝐀ₜ′ = 𝛅ₜ
        end
    end
        
    # update actor critic on entire data. Truncated backprop through time.
    function update_actor_critic_one_epoch!(actor, critic, data)
        ℓ, v̄, H̄, kl = 0, 0, 0, 0
        isrecurrent && set_rnn_state!.((actor, critic), init_𝐡ₜ)
        θ = Flux.params(actor, critic)
        batch_size = clamp(ppo.batch_size, N, N*M)
        batch_nsteps = batch_size ÷ N
        if !isrecurrent; data = shuffle(rng, data); end
        foreach(splitequal(M, batch_nsteps)) do timeindices
            ∇ = gradient(θ) do
                return mapfoldl(+, data[timeindices]) do (𝐬ₜ, 𝐚ₜ, 𝐫ₜ, 𝐝ₜ, 𝐭ₜ, 𝐬ₜ′, 𝛅ₜ, old𝛑ₜ, oldlog𝛑ₜ, 𝐯ₜ)
                    𝛑ₜ, log𝛑ₜ = get_probs_logprobs(actor, 𝐬ₜ)
                    if ppo.ppo
                        𝑟ₜ =  𝛑ₜ[𝐚ₜ] ./ old𝛑ₜ[𝐚ₜ]
                        actor_lossₜ = -min.(𝑟ₜ .* 𝛅ₜ, clamp.(𝑟ₜ, 1-ϵ, 1+ϵ) .* 𝛅ₜ) |> mean
                    else
                        actor_lossₜ = -𝛅ₜ .* log𝛑ₜ[𝐚ₜ] |> mean
                    end
                    critic_lossₜ = Flux.mse(critic(𝐬ₜ), 𝐯ₜ + 𝛅ₜ)
                    H̄ₜ = -sum(𝛑ₜ .* log𝛑ₜ; dims=1) |> mean
                    lossₜ = actor_lossₜ + critic_lossₜ - entropy_bonus * H̄ₜ

                    if isrecurrent
                        reset_idxs::BitVector = Zygote.@ignore (cpu(𝐝ₜ + 𝐭ₜ) .> 0)[1, :]
                        reset_rnn_state!.((actor, critic), (reset_idxs, reset_idxs));
                    end

                    H̄ += Zygote.@ignore H̄ₜ / M
                    ℓ += Zygote.@ignore lossₜ / M
                    v̄ += Zygote.@ignore mean(𝐯ₜ) / M
                    kl += Zygote.@ignore kldivergence(old𝛑ₜ, 𝛑ₜ) / M

                    return lossₜ
                end
            end
            Flux.update!(ppo.optim, θ, ∇)
        end
        return ℓ, v̄, H̄, kl
    end

    function calc_kl_div(actor, data)
        kl = 0
        isrecurrent && set_rnn_state!(actor, init_𝐡ₜ[1])
        return mapfoldl(+, data) do (𝐬ₜ, 𝐚ₜ, 𝐫ₜ, 𝐝ₜ, 𝐭ₜ, 𝐬ₜ′, 𝛅ₜ, old𝛑ₜ, oldlog𝛑ₜ, 𝐯ₜ)
            𝛑ₜ, log𝛑ₜ = get_probs_logprobs(actor, 𝐬ₜ)
            isrecurrent && reset_rnn_state!(actor, (cpu(𝐝ₜ + 𝐭ₜ) .> 0)[1, :]);
            return kldivergence(old𝛑ₜ, 𝛑ₜ) / M
        end
        return ℓ, v̄, H̄, kl
    end

    function update_actor_critic_with_early_stopping!(actor, critic, data, epochs)
        ℓ, v̄, H̄, kl = 0, 0, 0, 0
        num_epochs = 0
        for epoch in 1:epochs
            ℓ, v̄, H̄, kl = update_actor_critic_one_epoch!(actor, critic, data)
            num_epochs += 1
            kl = calc_kl_div(actor, data)
            kl >= kl_target && break
        end
        return ℓ, v̄, H̄, kl, num_epochs
    end

    data = collect_trajectories(ppo.actor, ppo.critic)
    update_advantates!(data)
    if ppo.device == gpu; data = device(map(dataₜ -> ppo.device.(dataₜ), data)); end;

    Flux.loadparams!(ppo.actor_gpu, Flux.params(ppo.actor))
    Flux.loadparams!(ppo.critic_gpu, Flux.params(ppo.critic))

    ℓ, v̄, H̄, kl, num_epochs = update_actor_critic_with_early_stopping!(ppo.actor_gpu, ppo.critic_gpu, data, nepochs)

    Flux.loadparams!(ppo.actor, Flux.params(ppo.actor_gpu))
    Flux.loadparams!(ppo.critic, Flux.params(ppo.critic_gpu))
    if isrecurrent
        set_rnn_state!.((ppo.actor, ppo.critic), get_rnn_state.((ppo.actor_gpu, ppo.critic_gpu)))
        ppo.𝐡ₜ = get_rnn_state.((ppo.actor, ppo.critic))
    end

    ppo.stats[:ℓ] = ℓ
    ppo.stats[:H̄] = H̄
    ppo.stats[:kl] = kl
    ppo.stats[:v̄] = v̄
    ppo.stats[:num_epochs] = num_epochs

    @debug "learning stats" steps episodes stats...
    nothing
end