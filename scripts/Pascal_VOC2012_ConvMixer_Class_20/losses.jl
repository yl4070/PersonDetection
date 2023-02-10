

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

function _segmentationloss(ypreds, ys; kwargs...)

    sz_preds = size(ypreds)
    ypreds = reshape(ypreds, :, sz_preds[end - 1], sz_preds[end])
    sz = size(ys)
    ys = reshape(ys, :, sz[end - 1], sz[end])
    Flux.Losses.logitcrossentropy(ypreds, ys; dims = 2, kwargs...)
end

function pointsentropy(preds, targets, pos_inds)

    ypreds = preds .* pos_inds 
    tot = _segmentationloss(ypreds, targets; agg = sum)

    n = sum(pos_inds)
    n = n == 0 ? 1 : n

    sum(tot) / n
end


function part_loss(ŷ, y)
 
    pos_idx = f32(y.heatmap .== 1)
    flt_idx = sum(pos_idx; dims = 3)
        
    Lk = loss_k(ŷ[1], y.heatmap, pos_idx)
    Lsz = pointsloss(ŷ[2], y.size, flt_idx)
    Loff = pointsloss(ŷ[3], y.off, flt_idx)

    Lk + Lsz + Loff
end


function tot_loss(ŷ, y1, y2, y3)

    ŷ1, ŷ2, ŷ3 = ŷ[1], ŷ[2][1], ŷ[2][2]
 

    part_loss(ŷ1, y1) + part_loss(ŷ2, y2) + part_loss(ŷ3, y3)
end