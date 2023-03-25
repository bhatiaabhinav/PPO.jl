using MDPs
import MDPs: preexperiment, postepisode, poststep
using UnPack
using Random
using Flux
using Flux.Zygote
using Flux: Optimiser, ClipNorm
import ProgressMeter: @showprogress, Progress, next!, finish!

export PPOLearner

Base.@kwdef mutable struct PPOLearner <: AbstractHook
    envs::Vector{AbstractMDP}   # A vector of differently seeded environments.
    actor::Union{Actor, RecurrentActor}       # `Actor` maps states to action probabilities, and the input size of dimensionality of state space. `RecurrentActor` maps catenation of [latest action (onehot), latest reward, current state] to action probabilities, and therefore the input size is: number of actions + 1 + dimensionality of states
    critic                      # Any Flux model mapping 𝑆 -> ℝ. If the actor is a `RecurrentActor`, the critic is also expected to be a recurrent model mapping ℝ^|𝐴| × ℝ × 𝑆 -> ℝ. i.e., like for the actor, the inputs are a catenation of [latest action (onehot), latest reward, current state].
    γ::Float32 = 0.99           # discount factor. Used to calulate TD(λ) advantages.
    nsteps::Int = 100           # numer of steps per iteration. So that total data per iteration = nenvs * nsteps. With longer nsteps, TD(λ) returns are computed better.
    nepochs::Int = 10           # number of epochs per iteration.
    minibatch_size::Int = 512
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
    ppo = true                  # whether to use PPO clip objective. If false, standard advantange actor-critic (A2C) objective will be used.
    device = cpu                # `cpu` or `gpu`
    progressmeter::Bool = false # Whether to show data and gradient updates progress using a progressmeter

    # data structures:
    optim = make_adam_optim(lr, (0.9, 0.999), adam_epsilon, clipnorm, adam_weight_decay)
    actor_gpu = device(deepcopy(actor))
    critic_gpu = device(deepcopy(critic))
    𝐡ₜ = nothing                # hidden states of the RNNs between training iterations
    𝐬ₜ = nothing                # states of envs between training iterations
    stats = Dict{Symbol, Any}()
end


function postepisode(ppo::PPOLearner; returns, steps, rng, kwargs...)
    @unpack envs, γ, nsteps, nepochs, minibatch_size, entropy_bonus, λ, ϵ, kl_target, device, progressmeter = ppo

    isrecurrent = ppo.actor isa RecurrentActor
    m = size(state_space(envs[1]), 1)
    n = size(action_space(envs[1]), 1)
    N = length(envs)
    M = nsteps
    B = N * M
    b = min(minibatch_size, B)
    episodes = length(returns)

    if isnothing(ppo.𝐬ₜ)
        Threads.@threads for i in 1:N; reset!(envs[i], rng=rng); end
        𝐬ₜ = mapfoldl(state, hcat, envs) |> tof32
        if isrecurrent
            𝐬ₜ = vcat(zeros(Float32, n, N), zeros(Float32, 1, N), 𝐬ₜ)
        end
        init_𝐬ₜ = 𝐬ₜ
        reset_rnn_state!.((ppo.actor, ppo.critic))
        init_𝐡ₜ = get_rnn_state.((ppo.actor, ppo.critic))
    else
        init_𝐬ₜ = ppo.𝐬ₜ
        init_𝐡ₜ = ppo.𝐡ₜ
    end

    function collect_trajectories(actor, critic)    
        𝐬ₜ = init_𝐬ₜ # start where we left off
        isrecurrent && set_rnn_state!.((actor, critic), init_𝐡ₜ)
        𝐯ₜ = critic(𝐬ₜ)
        data = []
        progress = Progress(M; color=:blue, desc="Collecting trajectories", enabled=progressmeter)
        for t in 1:M
            𝛑ₜ, log𝛑ₜ = get_probs_logprobs(actor, 𝐬ₜ)   # Forward actor
            𝐚ₜ = zeros(Int, 1, N)
            Threads.@threads for i in 1:N
                aᵢₜ = sample(rng, 1:n, ProbabilityWeights(𝛑ₜ[:, i]))
                step!(envs[i], aᵢₜ; rng=rng)
                𝐚ₜ[1, i] = aᵢₜ
            end
            𝐫ₜ = mapfoldl(reward, hcat, envs) |> tof32
            𝐬ₜ′= mapfoldl(state, hcat, envs) |> tof32
            𝐝ₜ = mapfoldl(in_absorbing_state, hcat, envs) |> tof32
            𝐭ₜ = mapfoldl(truncated, hcat, envs) |> tof32
            if isrecurrent
                𝐚ₜ_onehot = @views Flux.OneHotArrays.onehotbatch(𝐚ₜ[1, :], 1:n) |> tof32
                𝐬ₜ′ = vcat(𝐚ₜ_onehot, 𝐫ₜ, 𝐬ₜ′)
                𝐡ₜ_backup = get_rnn_state(critic)  # create a backup
            end
            𝐯ₜ′ = critic(𝐬ₜ′)
            𝛅ₜ = 𝐫ₜ + γ * (1f0 .- 𝐝ₜ) .* 𝐯ₜ′ - 𝐯ₜ
            dataₜ = (𝐬ₜ, 𝐚ₜ, 𝐫ₜ, 𝐝ₜ, 𝐭ₜ, 𝐬ₜ′, 𝛅ₜ, 𝛑ₜ, log𝛑ₜ, 𝐯ₜ)
            push!(data, dataₜ)
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
            next!(progress)
        end
        ppo.𝐬ₜ = 𝐬ₜ
        finish!(progress)
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

    function flattened(data)
        𝐬, 𝐚, 𝐫, 𝐝, 𝐭, 𝐬′, 𝛅, 𝛑, log𝛑, 𝐯 = map(i -> mapfoldl(d -> d[i], hcat, data), 1:length(data[1]))
        return 𝐬, 𝐚, 𝐫, 𝐝, 𝐭, 𝐬′, 𝛅, 𝛑, log𝛑, 𝐯
    end


    function to_gpu(data)
        𝐬, 𝐚, 𝐫, 𝐝, 𝐭, 𝐬′, 𝛅, 𝛑, log𝛑, 𝐯 = data
        𝐬, 𝐫, 𝐬′, 𝛅, 𝛑, log𝛑, 𝐯 = gpu.((𝐬, 𝐫, 𝐬′, 𝛅, 𝛑, log𝛑, 𝐯))
        return 𝐬, 𝐚, 𝐫, 𝐝, 𝐭, 𝐬′, 𝛅, 𝛑, log𝛑, 𝐯
    end

    function ppo_loss(actor, critic, 𝐬, 𝐚, 𝛅, old𝛑, 𝐯)
        # ---- critic loss ----
        critic_loss = Flux.mse(critic(𝐬), 𝐯 + 𝛅)
        # ---- actor loss ----
        𝐚 = Zygote.@ignore CartesianIndex.(zip(𝐚, (1:size(𝐯, 2))'))
        𝛑, log𝛑 = get_probs_logprobs(actor, 𝐬)
        𝛅 = !ppo.center_advantages ? 𝛅 : Zygote.@ignore 𝛅 .- mean(𝛅) 
        𝛅 = !ppo.scale_advantages ? 𝛅 : Zygote.@ignore 𝛅 ./ (std(𝛅) + 1e-8)
        if ppo.ppo
            𝑟 =  𝛑[𝐚] ./ old𝛑[𝐚]
            actor_loss = -min.(𝑟 .* 𝛅, clamp.(𝑟, 1-ϵ, 1+ϵ) .* 𝛅) |> mean
        else
            actor_loss = -𝛅 .* log𝛑[𝐚] |> mean
        end
        # ---- entropy bonus ----
        entropy = -sum(𝛑 .* log𝛑; dims=1) |> mean
        # ---- total loss ----
        return actor_loss + critic_loss - entropy_bonus * entropy
    end

    function update_actor_critic_one_epoch_recurrent!(actor, critic, data)
        loss, θ = 0f0, Flux.params(actor, critic)
        𝐬, 𝐚, _, 𝐝, 𝐭, _, 𝛅, old𝛑, _, 𝐯 = data
        𝐬, 𝐚, 𝐝, 𝐭, 𝛅, old𝛑, 𝐯 = map(𝐱 -> reshape(𝐱, :, N, M), (𝐬, 𝐚, 𝐝, 𝐭, 𝛅, old𝛑, 𝐯))    # reshape to 3D to make time as last axis.
        nsgdsteps = ceil(Int, B / b)
        _N = ceil(Int, N / nsgdsteps) # num envs per minibatch
        progress = Progress(N; desc="Performing gradient updates", color=:blue, enabled=progressmeter)
        for env_indices in splitequal(N, _N)
            _𝐬, _𝐚, _𝐝, _𝐭, _𝛅, _old𝛑, _𝐯 = map(𝐱 -> 𝐱[:, env_indices, :], (𝐬, 𝐚, 𝐝, 𝐭, 𝛅, old𝛑, 𝐯))
            set_rnn_state!.((actor, critic), init_𝐡ₜ, (env_indices, env_indices))
            _loss, _∇ =  withgradient(θ) do
                return mapfoldl(+, 1:M) do t
                    _𝐬ₜ, _𝐚ₜ, _𝐝ₜ, _𝐭ₜ, _𝛅ₜ, _old𝛑ₜ, _𝐯ₜ = Zygote.@ignore @views map(𝐱 -> 𝐱[:, :, t], (_𝐬, _𝐚, _𝐝, _𝐭, _𝛅, _old𝛑, _𝐯))
                    _lossₜ = ppo_loss(actor, critic, _𝐬ₜ, _𝐚ₜ, _𝛅ₜ, _old𝛑ₜ, _𝐯ₜ)
                    if isrecurrent
                        reset_idxs::BitVector = Zygote.@ignore ((_𝐝ₜ + _𝐭ₜ) .> 0)[1, :]
                        reset_rnn_state!.((actor, critic), (reset_idxs, reset_idxs));
                    end
                    return _lossₜ / M
                end
            end
            Flux.update!(ppo.optim, θ, _∇)
            loss += _loss * length(env_indices) / N
            next!(progress; step=length(env_indices))
        end
        finish!(progress)
        return loss
    end

    function update_actor_critic_one_epoch!(actor, critic, data)
        loss, θ = 0f0, Flux.params(actor, critic)
        𝐬, 𝐚, _, _, _, _, 𝛅, old𝛑, _, 𝐯 = data
        nsgdsteps = ceil(Int, B / b)
        progress = Progress(nsgdsteps; desc="Performing gradient updates", color=:blue, enabled=progressmeter)
        for i in 1:nsgdsteps
            mb_indices = rand(rng, 1:B, b)
            _𝐬, _𝐚, _𝛅, _old𝛑, _𝐯 = map(𝐱 -> 𝐱[:, mb_indices], (𝐬, 𝐚, 𝛅, old𝛑, 𝐯))
            _loss, _∇ = withgradient(θ) do
                return ppo_loss(actor, critic, _𝐬, _𝐚, _𝛅, _old𝛑, _𝐯)
            end
            Flux.update!(ppo.optim, θ, _∇)
            loss += _loss / nsgdsteps
            next!(progress)
        end
        finish!(progress)
        return loss
    end

    # function update_actor_critic_one_epoch_recurrent_truncated_bptt!(actor, critic, data)
    #     loss = 0
    #     𝐬, 𝐚, _, 𝐝, 𝐭, _, 𝛅, old𝛑, _, 𝐯 = data
    #     𝐬, 𝐚, 𝐝, 𝐭, 𝛅, old𝛑, 𝐯 = @views map(𝐱 -> reshape(𝐱, :, N, M), (𝐬, 𝐚, 𝐝, 𝐭, 𝛅, old𝛑, 𝐯))    # reshape to 3D to make time as last axis.
    #     set_rnn_state!.((actor, critic), init_𝐡ₜ)
    #     θ = Flux.params(actor, critic)
    #     _M = ceil(Int, b/N)
    #     foreach(splitequal(M, _M)) do timeindices
    #         _loss, ∇ = withgradient(θ) do
    #             return mapfoldl(+, timeindices) do t
    #                 (𝐬ₜ, 𝐚ₜ, 𝐝ₜ, 𝐭ₜ, 𝛅ₜ, old𝛑ₜ, 𝐯ₜ) = Zygote.@ignore @views map(𝐱 -> 𝐱[:, :, t], (𝐬, 𝐚, 𝐝, 𝐭, 𝛅, old𝛑, 𝐯))
    #                 lossₜ = ppo_loss(actor, critic, 𝐬ₜ, 𝐚ₜ, 𝛅ₜ, old𝛑ₜ, 𝐯ₜ)
    #                 if isrecurrent
    #                     reset_idxs::BitVector = Zygote.@ignore (cpu(𝐝ₜ + 𝐭ₜ) .> 0)[1, :]
    #                     reset_rnn_state!.((actor, critic), (reset_idxs, reset_idxs));
    #                 end
    #                 return lossₜ / M
    #             end
    #         end
    #         loss += _loss
    #         Flux.update!(ppo.optim, θ, ∇)
    #     end
    #     return loss
    # end

    function calculate_stats(actor, critic, data)
        H̄, v̄, kl = 0, 0, 0
        𝐬, _, _, 𝐝, 𝐭, _, _, old𝛑, _, _ = data
        𝐬, 𝐝, 𝐭, old𝛑 = map(𝐱 -> reshape(𝐱, :, N, M), (𝐬, 𝐝, 𝐭, old𝛑))
        isrecurrent && set_rnn_state!.((actor, critic), init_𝐡ₜ)
        for t in 1:M
            𝐬ₜ, 𝐝ₜ, 𝐭ₜ, old𝛑ₜ = map(𝐱 -> 𝐱[:, :, t], (𝐬, 𝐝, 𝐭, old𝛑))
            𝛑ₜ, log𝛑ₜ = get_probs_logprobs(actor, 𝐬ₜ)
            H̄ += -sum(𝛑ₜ .* log𝛑ₜ; dims=1) |> mean
            kl += kldivergence(old𝛑ₜ, 𝛑ₜ)
            v̄ += critic(𝐬ₜ) |> mean
            if isrecurrent
                reset_idxs::BitVector = ((𝐝ₜ + 𝐭ₜ) .> 0)[1, :]
                reset_rnn_state!.((actor, critic), (reset_idxs, reset_idxs));
            end
        end
        return (kl, H̄, v̄) ./ M
    end

    function update_actor_critic_with_early_stopping!(actor, critic, data, epochs)
        ℓ, kl, H̄, v̄ = 0, 0, 0, 0
        num_epochs = 0
        update_fn = isrecurrent ? update_actor_critic_one_epoch_recurrent! : update_actor_critic_one_epoch!
        for epoch in 1:epochs
            ℓ = update_fn(actor, critic, data)
            num_epochs += 1
            kl, H̄, v̄ = calculate_stats(actor, critic, data)
            kl >= kl_target && break
        end
        return ℓ, kl, H̄, v̄, num_epochs
    end

    data = collect_trajectories(ppo.actor, ppo.critic)
    update_advantates!(data)
    data = flattened(data)

    if ppo.device == gpu; data = to_gpu(data); end;

    Flux.loadparams!(ppo.actor_gpu, Flux.params(ppo.actor))
    Flux.loadparams!(ppo.critic_gpu, Flux.params(ppo.critic))

    ℓ, kl, H̄, v̄, num_epochs = update_actor_critic_with_early_stopping!(ppo.actor_gpu, ppo.critic_gpu, data, nepochs)
    
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