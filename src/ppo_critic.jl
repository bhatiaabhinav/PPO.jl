
"""critic can be any flux model"""
function get_values(critic, 𝐬, recurtype::RecurrenceType)
    if recurtype ∈ (MARKOV, TRANSFORMER) || ndims(𝐬) == 2
        return critic(𝐬)
    else
        # interpret as (state_dim, ntimesteps, batch_size) for the RNN
        Flux.Zygote.@ignore Flux.reset!(critic)
        mapfoldl(hcat, 1:size(𝐬, 2)) do t
            return reshape(critic(𝐬[:, t, :]), 1, 1, :)
        end
    end
end

function get_critic_loss(critic, 𝐬, 𝐯, 𝛅, recurtype::RecurrenceType)
    𝐯̂ = get_values(critic, 𝐬, recurtype)
    Flux.mse(𝐯̂, 𝐯 + 𝛅)
end