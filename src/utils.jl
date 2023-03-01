function tof32(𝐱::AbstractArray{<:Real, N})::AbstractArray{Float32, N} where N
    convert(AbstractArray{Float32, N}, 𝐱)
end


const SQRT_2PI = Float32(sqrt(2π))
const LOG_SQRT_2PI = Float32(log(SQRT_2PI))

@inline function log_nomal_prob(x::T, μ::T, σ::T)::T where {T<:Real}
    return -0.5 * ((x - μ) / σ)^2 - log(SQRT_2PI * σ)
end

@inline function log_nomal_prob(x::T, μ::T, σ::T, logσ::T)::T where {T<:Real}
    return -0.5 * ((x - μ) / σ)^2 - LOG_SQRT_2PI - logσ
end

function splitequal(H, TH)
    splits = []
    t = 0
    while t < H
        push!(splits, (t+1):min((t+TH), H))
        t += TH
    end
    return splits
end



using Flux



function make_adam_optim(lr, betas, epsilon, clipnorm, weight_decay)
    adam = Adam(lr, betas, epsilon)
    if weight_decay > 0
        adam = Optimiser(WeightDecay(weight_decay), adam)
    end
    if clipnorm < Inf
        adam = Optimiser(ClipNorm(clipnorm), adam)
    end
    return adam
end



get_rnn_state(m::Flux.Recur) = m.state
get_rnn_state(m) = map(get_rnn_state, Flux.functor(m)[1])


function set_rnn_state!(m::Flux.Recur, state)
    if m.state isa Flux.CUDA.CuArray
        state = gpu(state)
    else
        state = cpu(state)
    end
    m.state = state
    nothing
end

function set_rnn_state!(m, state)
    foreach(zip(Flux.functor(m)[1], state)) do (_m, _state)
        set_rnn_state!(_m, _state)
    end
    nothing
end


function reset_rnn_state!(rnn)
    Flux.reset!(rnn)
    nothing
end

"""reset rnn state for just idx-th index in the batch. idx can also be a bool array marking all the indices for which the rnn sttae needs to be reset."""
function reset_rnn_state!(m::Flux.Recur, idx::Union{Int, BitVector})
    if (idx isa BitArray) && all(idx)
        reset_rnn_state!(m)
        return nothing
    end
    if (idx isa BitArray) && !any(idx)
        return nothing
    end
    n = size(m.state)[end]  # batch size
    mask = Flux.Zygote.ignore() do
        mask = zeros(eltype(m.state), 1, n)
        mask[1, idx] .= 1
        if m.state isa Flux.CUDA.CuArray
            mask = gpu(mask)
        end
        return mask
    end
    m.state = m.state .* (1 .- mask) + m.cell.state0 * mask
    nothing
end
function reset_rnn_state!(m, idx::Union{Int, BitVector})
    if (idx isa BitArray) && all(idx)
        reset_rnn_state!(m)
        return nothing
    end
    if (idx isa BitArray) && !any(idx)
        return nothing
    end
    foreach(_m -> reset_rnn_state!(_m, idx), Flux.functor(m)[1])
    nothing
end
