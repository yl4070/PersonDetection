


function loss_k(preds, targets, pos_inds)
    
    α = 2f0
    β = 4f0
    neg_inds = 1 .- pos_inds

    neg_weights = @. (1 - targets)^β

    preds = clamp.(preds, 1e-4, 1-1e-4)
    pos_loss = @. log(preds) * (1-preds)^α * pos_inds
    neg_loss = @. log(1-preds) * preds^α * neg_weights * neg_inds

    num_pos = sum(pos_inds)
    pos_loss = sum(pos_loss)
    neg_loss = sum(neg_loss)

    if num_pos == 0 
        - neg_loss
    else
        (- pos_loss - neg_loss) / num_pos
    end

end

function pointsloss(preds, targets, pos_inds)

    n = sum(pos_inds)
    n = n == 0 ? 1 : n
    tot = @. ((preds - targets) * pos_inds) |> Flux.abs2

    sum(tot) / n
end


# ANCHOR loss function for evdiential

using SpecialFunctions

function kl2(α)
    
    ψ = SpecialFunctions.digamma
    lnΓ = SpecialFunctions.loggamma

    K = size(α)[3] 
    ∑α = sum(α, dims = 3)
    ∑lnΓα = sum(lnΓ.(α), dims = 3)

    A = @. lnΓ(∑α) - lnΓ(K) - ∑lnΓα
    B = sum(@.( (α-1) * (ψ(α) - ψ(∑α)) ), dims = 3)

    A+B
end

function convdirloss(α, y, t)
    
    S = sum(α, dims = 3)

    p̂ = α ./ S

    loss = (y - p̂).^2 .+ p̂ .* (1 .- p̂) ./ (S .+ 1)
    loss = sum(loss; dims = 3) 

    λ = min(1., t / 10)
    α̂ = @. y + (1-y) * α
    reg = kl2(α̂)

    sum(loss .+ λ .* reg)
end

function pointsdir(α, y, t, pos_inds)

    n = sum(pos_inds)
    n = n == 0 ? 1 : n

    S = sum(α, dims = 3)

    p̂ = α ./ S

    loss = (y - p̂).^2 .+ p̂ .* (1 .- p̂) ./ (S .+ 1)
    loss = sum(loss; dims = 3) # reduce the channel to 1

    λ = min(1., t / 10)
    α̂ = @. y + (1-y) * α
    reg = kl2(α̂)
    
    tot = @. (loss + λ * reg) * pos_inds

    sum(tot) / n
end



# ANCHOR loss for NIG2

function nllstudent2(y, γ, ν, α, β)
    Ω = @. 2β * (1 + ν)
    logγ = SpecialFunctions.loggamma
    nll = 0.5 * log.(π ./ ν) -
          α .* log.(Ω) +
          (α .+ 0.5) .* log.(ν .* (y - γ) .^ 2 + Ω) +
          logγ.(α) -
          logγ.(α .+ 0.5)

    nll
end

using Statistics

aleatoric(ν, α, β) = @. (β * (1 + ν)) / (ν * α)

function nigloss2(o, y, λ=1, p=1, n=2)

    γ = o[:,:,1:n,:]
    ν = o[:,:,n+1:2n,:] .|> Flux.softplus
    α = Flux.softplus(o[:,:,2n+1:3n,:]) .+ 1
    β = o[:,:,3n+1:4n,:] .|> Flux.softplus

    nll = nllstudent2(y, γ, ν, α, β)

    uₐ = aleatoric(ν, α, β)

    error = @. (abs(y - γ) / uₐ)^p
    Φ = evidence(ν, α)
    reg = error .* Φ

    loss = @. nll + λ * reg

    mean(loss; dims = 3)
end


function pointnig(nigloss, pos_idx)

    n = sum(pos_idx)
    n = n == 0 ? 1 : n

    tot = nigloss .* pos_idx

    sum(tot) / n
end


o = rand(Float32, 64, 64, 8, 10)
pointnig(nigloss2(y.size, o), pos_idx)








