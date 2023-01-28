
# function nonMaxSupression!(detections::AbstractVector, iou_thresh::T) where {T<:AbstractFloat}
#     sort!(detections, by = x -> x.conf, rev = true)
#     for i = 1:lastindex(detections)
#         k = i + 1
#         @views while k <= length(detections)
#             iou = ioumatch(detections[i], detections[k])
#             if iou > iou_thresh # && (detections[i].class == detections[k].class)
#                 deleteat!(detections, k)
#                 k -= 1
#             end
#             k += 1
#         end
#     end
# end

function nonMaxSupression(detections::AbstractVector, iou_thresh::T) where {T<:AbstractFloat}

    for idx in axes(detections, 1)

        iou = [bboxiou(detections[idx], detections[j]) for j in idx+1:length(detections)]

        same_objects = findall(>=(iou_thresh), iou)
        detections = detections[setdiff(1:length(detections), same_objects .+ idx)]
        idx >= length(detections) && break
    end

    detections
end

function bboxiou(box1, box2)
    b1x1, b1y1, b1x2, b1y2 = box1.x, box1.y, box1.x+box1.w, box1.y+box1.h
    b2x1, b2y1, b2x2, b2y2 = box2.x, box2.y, box2.x+box2.w, box2.y+box2.h
    rectx1 = max(b1x1, b2x1)
    recty1 = max(b1y1, b2y1)
    rectx2 = min(b1x2, b2x2)
    recty2 = min(b1y2, b2y2)

    interarea = max(rectx2 - rectx1, 0) * max(recty2 - recty1, 0)
    b1area = (b1x2 - b1x1) * (b1y2 - b1y1)
    b2area = (b2x2 - b2x1) * (b2y2 - b2y1)
    iou = interarea / (b1area + b2area - interarea)
    return iou
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
    # yhat[4]

    idx = findall(el -> el >= 0.3, yhat[1][:,:,1,1])
    # idx_ch = findall(el -> el <= -0.3, yhat[1][:,:,1,1])

    # uncerns = map(idx) do i
    #     uncertainty(yhat[4][i, :, 1]) # evidential
    # end
    # class = map(idx) do id
    #     Flux.onecold(yhat[4][id, :, 1], classes)
    # end
    confs = map(idx) do id
        yhat[1][id, 1, 1] # heatmap
    end
    sizevs = map(idx) do id
        yhat[2][id, 1, 1]
    end
    sizehs = map(idx) do id
        yhat[2][id, 2, 1]
    end

    # TODO add uncertainty for sizes

    offvs = map(idx) do id
        yhat[3][id, 1, 1]
    end
    offhs = map(idx) do id
        yhat[3][id, 2, 1]
    end

    detections = []
    for (id, szv, szh, offv, offh, conf) in zip(idx, sizevs, sizehs, offvs, offhs, confs)
        
        v, h = id[1], id[2]

        x, y = (h + offh, v + offv)
        x, y = (4x-szh/2, 4y-szv/2) .|> round .|> Int

        szh, szv = (szh, szv) .|> round .|> Int

        # draw!(img, Polygon(RectanglePoints(ImageDraw.Point(x, y), ImageDraw.Point(x+szh, y+szv))), colorant"yellow")
        push!(detections, (x=x, y=y, w=szh, h=szv, conf=conf, uncern = uncern, class = cls))
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


