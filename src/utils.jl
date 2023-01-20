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
        push!(splits, (t+1):(t+TH))
        t += TH
    end
    return splits
end