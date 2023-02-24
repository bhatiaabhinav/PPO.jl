using MDPs
import MDPs: preexperiment, postepisode, poststep
using UnPack
using Random
using Flux
using Flux.Zygote
using Flux: Optimiser, ClipNorm

export PPOLearner

Base.@kwdef struct PPOLearner <: AbstractHook
    envs::Vector{AbstractMDP}
    actor
    critic
    γ::Float32 = 0.99
    horizon::Int
    ent_bonus::Float32 = 0.01f0
    clipnorm = Inf
    actor_lr = 0.0003
    critic_lr = 0.001
    optim_actor = Adam(actor_lr)
    optim_critic = clipnorm < Inf ? Optimiser(ClipNorm(clipnorm), Adam(critic_lr)) : Adam(critic_lr)
    train_interval = horizon
    tbptt_horizon = 4
    gradsteps = 1
    λ = 0.95f0
    ϵ = 0.2f0
    kl_target = 0.01
    max_actor_iters = 20
    ppo = true
    device = cpu
    actor_gpu = device(actor)
    critic_gpu = device(critic)
    stats = Dict{Symbol, Any}()
end


function poststep(a2cl::PPOLearner; returns, steps, rng, kwargs...)
    @unpack actor, critic, actor_gpu, critic_gpu, envs, horizon, γ, ent_bonus, optim_actor, optim_critic, train_interval, tbptt_horizon, gradsteps, λ, ϵ, kl_target, max_actor_iters, stats, ppo, device = a2cl
    isrecurrent = actor isa RecurrentActor

    n = size(action_space(envs[1]), 1)
    nenvs = length(envs)

    function train_actor_critic()

        # collect data using synchronous parallel actors
        function collect_trajectories()
            Flux.reset!(actor)
            Flux.reset!(critic)
            for env in envs
                reset!(env; rng=rng)
            end
            𝐬ₜ = Zygote.@ignore mapreduce(state, hcat, envs) |> tof32
            if isrecurrent
                𝐬ₜ = Zygote.@ignore vcat(zeros(Float32, n, nenvs), zeros(Float32, 1, nenvs), 𝐬ₜ, zeros(Float32, 1, nenvs))
            end
            𝐯ₜ = critic(𝐬ₜ)
            data = map(1:horizon) do t
                𝛑ₜ, _ = get_probs_logprobs(actor, 𝐬ₜ)
                𝐚ₜ = mapreduce(hcat, 1:nenvs) do i  # 1-TODO: Can be parallelized
                    aᵢₜ = sample(rng, 1:n, ProbabilityWeights(𝛑ₜ[:, i]))
                    in_absorbing_state(envs[i]) ? reset!(envs[i]; rng=rng) : step!(envs[i], aᵢₜ; rng=rng)
                    return CartesianIndex(aᵢₜ, i)
                end
                𝐫ₜ = mapreduce(reward, hcat, envs) |> tof32
                𝐬ₜ′= mapreduce(state, hcat, envs) |> tof32
                𝐝ₜ = mapreduce(in_absorbing_state, hcat, envs) |> tof32
                if isrecurrent
                    aonehot = zeros(Float32, n, nenvs)
                    aonehot[𝐚ₜ] .= 1
                    𝐬ₜ′ = vcat(aonehot, 𝐫ₜ, 𝐬ₜ′, 𝐝ₜ)
                end
                𝐯ₜ′ = critic(𝐬ₜ′)
                # @info "hey" nenvs size(𝐫ₜ) size(𝐝ₜ) size(𝐯ₜ) size(𝐯ₜ′)
                𝛅ₜ = 𝐫ₜ + γ * (1f0 .- 𝐝ₜ) .* 𝐯ₜ′ - 𝐯ₜ
                dataₜ = (𝐬ₜ, 𝐚ₜ, 𝐫ₜ, 𝐝ₜ, 𝐬ₜ′, 𝛅ₜ, 𝛑ₜ, 𝐯ₜ)
                𝐬ₜ = 𝐬ₜ′
                𝐯ₜ = 𝐯ₜ′
                return dataₜ
            end
            return data
        end
        
        # update advantages using GAE
        function update_advantates!(data)
            𝐀ₜ′ = 0
            for dataₜ in reverse(data)
                (𝐬ₜ, 𝐚ₜ, 𝐫ₜ, 𝐝ₜ, 𝐬ₜ′, 𝛅ₜ, 𝛑ₜ, 𝐯ₜ) = dataₜ
                𝛅ₜ .+= γ * λ * (1f0 .- 𝐝ₜ) .* 𝐀ₜ′
                𝐀ₜ′ = 𝛅ₜ
            end
        end
        
        # update critic. Truncated backprop through time
        function update_critic!(critic, data)
            v̄ = 0
            ϕ = Flux.params(critic)
            Flux.reset!(critic);
            ℓϕ = mapreduce(+, splitequal(horizon, tbptt_horizon)) do timechunk
                loss, ∇ = withgradient(ϕ) do
                    critic_loss = 0f0
                    for (𝐬ₜ, 𝐚ₜ, 𝐫ₜ, 𝐝ₜ, 𝐬ₜ′, 𝛅ₜ, 𝛑ₜ, 𝐯ₜ) in data[timechunk]
                        critic_loss += Flux.mse(critic(𝐬ₜ), 𝐯ₜ + 𝛅ₜ)
                        v̄ += mean(𝐯ₜ)
                    end
                    return critic_loss
                end
                Flux.update!(optim_critic, ϕ, ∇)
                return loss
            end
            return (ℓϕ, v̄) ./ horizon
        end

        # update actor. Truncated backprop through time. Uses PPO-clip objective if `ppo` is true
        function update_actor!(actor, data)
            H̄, kl_div = 0, 0
            θ = Flux.params(actor)
            # update actor. Need a full forward rollout if want to handle recurrent network
            Flux.reset!(actor)
            ℓθ = mapreduce(+, splitequal(horizon, tbptt_horizon)) do timechunk
                loss, ∇ = withgradient(θ) do
                    actor_loss = 0f0
                    for (𝐬ₜ, 𝐚ₜ, 𝐫ₜ, 𝐝ₜ, 𝐬ₜ′, 𝐀ₜ, old𝛑ₜ, 𝐯ₜ) in data[timechunk]
                        𝛑ₜ, log𝛑ₜ = get_probs_logprobs(actor, 𝐬ₜ)
                        if ppo
                            𝑟ₜ = 𝛑ₜ[𝐚ₜ] ./ old𝛑ₜ[𝐚ₜ]
                            actor_loss += -min.(𝑟ₜ .* 𝐀ₜ, clamp.(𝑟ₜ, 1-ϵ, 1+ϵ) .* 𝐀ₜ) |> mean
                        else
                            actor_loss -= 𝐀ₜ .* log𝛑ₜ[𝐚ₜ] |> mean
                        end
                        H̄ₜ = -sum(𝛑ₜ .* log𝛑ₜ; dims=1) |> mean
                        actor_loss -= ent_bonus * H̄ₜ
                        H̄ += H̄ₜ
                        kl_div += kldivergence(old𝛑ₜ, 𝛑ₜ)
                    end
                    return actor_loss
                end
                Flux.update!(optim_actor, θ, ∇)
                return loss
            end
            return (ℓθ, H̄, kl_div) ./ horizon
        end

        function update_actor_multiple_iters!(actor, data, iters)
            (ℓθ, H̄, kl_div) = 0, 0, 0
            num_iters = 0
            for iter in 1:iters
                ℓθ, H̄, kl_div = update_actor!(actor, data)
                num_iters += 1
                kl_div >= kl_target && break
            end
            return ℓθ, H̄, kl_div, num_iters
        end

        data = collect_trajectories()
        update_advantates!(data)
        if device == gpu; data = map(dataₜ -> device.(dataₜ), data); end
        if actor_gpu.actor_model !== actor.actor_model; Flux.loadparams!(actor_gpu.actor_model, Flux.params(actor.actor_model)); end
        if critic_gpu !== critic; Flux.loadparams!(critic_gpu, Flux.params(critic)); end
        ℓθ, H̄, kl_div, num_iters = update_actor_multiple_iters!(actor_gpu, data, max_actor_iters)
        ℓϕ, v̄ = update_critic!(critic_gpu, data)
        if actor_gpu.actor_model !== actor.actor_model; Flux.loadparams!(actor.actor_model, Flux.params(actor_gpu.actor_model)); end
        if critic_gpu !== critic; Flux.loadparams!(critic, Flux.params(critic_gpu)); end
        return ℓθ, ℓϕ, H̄, kl_div, num_iters, v̄
    end

    episodes = length(returns)
    if steps % train_interval == 0
        for gradstep in 1:gradsteps
            ℓθ, ℓϕ, H, kl, num_actor_iters, v̄ = train_actor_critic()
            stats[:ℓθ] = ℓθ
            stats[:ℓϕ] = ℓϕ
            stats[:H] = H
            stats[:kl] = kl
            stats[:v̄] = v̄
            stats[:num_actor_iters] = num_actor_iters
        end
    end
    
    @debug "learning stats" steps episodes stats...
    nothing
end