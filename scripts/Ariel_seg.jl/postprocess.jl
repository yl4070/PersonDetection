
function nonMaxSupression!(detections::AbstractVector, iou_thresh::T) where {T<:AbstractFloat}
    sort!(detections, by = x -> x.conf, rev = true)
    for i = 1:lastindex(detections)
        k = i + 1
        @views while k <= length(detections)
            iou = ioumatch(detections[i], detections[k])
            if iou > iou_thresh # && (detections[i].class == detections[k].class)
                deleteat!(detections, k)
                k -= 1
            end
            k += 1
        end
    end
end

# Calculates IoU score (overlapping rate)

function ioumatch(bbox1, bbox2)
    r1 = bbox1.x + bbox1.w
    l1 = bbox1.x
    t1 = bbox1.y
    b1 = bbox1.y + bbox1.h
    r2 = bbox2.x + bbox2.w
    l2 = bbox2.x
    t2 = bbox2.y
    b2 = bbox2.y + bbox2.h
    a = min(r1, r2)
    b = max(t1, t2)
    c = max(l1, l2)
    d = min(b1, b2)
    intersec = (d - b) * (a - c)
    return intersec / (bbox1.w * bbox1.h + bbox2.w * bbox2.h - intersec)
end


function getprediction(model, imgx)


    yhat = model(imgx)

    idx = findall(el -> el > 0.3, yhat[3][:,:,1,1])

    uncerns = map(idx) do i
        uncertainty(yhat[4][i, :, 1]) # evidential
    end
    confs = map(idx) do id
        yhat[3][id, 1, 1] # heatmap
    end
    sizevs = map(idx) do id
        yhat[2][id, 1, 1]
    end
    sizehs = map(idx) do id
        yhat[2][id, 2, 1]
    end
    offvs = map(idx) do id
        yhat[1][id, 1, 1]
    end
    offhs = map(idx) do id
        yhat[1][id, 2, 1]
    end

    detections = []
    for (id, szv, szh, offv, offh, conf, uncern) in zip(idx, sizevs, sizehs, offvs, offhs, confs, uncerns)
        
        v, h = id[1], id[2]

        x, y = (h + offh, v + offv)
        x, y = (4x-szh/2, 4y-szv/2) .|> round .|> Int

        szh, szv = (szh, szv) .|> round .|> Int

        # draw!(img, Polygon(RectanglePoints(ImageDraw.Point(x, y), ImageDraw.Point(x+szh, y+szv))), colorant"yellow")
        push!(detections, (x=x, y=y, w=szh, h=szv, conf=conf, uncern = uncern))
    end

    detections
end




# using JSON3

# JSON3.@pretty detections1

# detections = Dict()

# for imgname in rand(names, 10)

#     img, imgx = getimgencode(imgname)
#     preds = getprediction(model, imgx)

#     detections[imgname] = preds
# end

# open("detections.json", "w") do io
#     JSON3.pretty(io, detections)
# end


