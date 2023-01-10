using Images
using MLUtils
using FastAI, FastVision

V, H = 64, 64

function cal_offset(v, h, vw, hw, R = 4)
    p = [v + vw÷2, h + hw÷2]
        
    p̃ = p .÷ R
    @. p / R - p̃
end

# cal_offset(boxes["006"][1]...)
function centerize(box, R = 4)
    v, h, vw, hw = box
    p = [v + vw÷2, h + hw÷2]

    check_bounds.(p .÷ R)
end

function expandtruth(boxes, ys, dim)

    p̃s = centerize.(boxes)

    L = zeros(dim)    
    for (p̃, y) in zip(p̃s, ys)
        
        L[p̃..., :] = y
    end
    L
end


# size = 256x256
Yxyc(v, h, pv, ph) = exp(-((v-pv)^2 + (h-ph)^2)/2)
function heatmap(boxes)

    Y = zeros(V, H)

    av, ah = axes(Y, 1), axes(Y, 2)

    for box in boxes

        Y′ = zeros(V, H)
        p̃ = centerize(box)

        for i in av, j in ah
            Y′[i,j] = Yxyc(i, j, p̃[1], p̃[2])
        end

        Y = max.(Y, Y′)
    end

    Y
end

expandim(x, d = 3) = Flux.unsqueeze(x, d)

# ANCHOR - get label data
names = keys(bbox_dict) |> collect

name = names[1]
yiter = mapobs(names) do name

    img_boxes = map(bbox_dict[name]) do b
        b.bbox
    end

    off = map(img_boxes) do box
        cal_offset(box...)
    end

    sizeL = map(img_boxes) do box
        box[3:4]
    end

    # TODO - Need category
    # REVIEW - all classes for now, 80
    class = map(bbox_dict[name]) do obj
        obj.cat_id |> id -> Flux.onehot(id, cat_ids)
    end

    # return named-tuple
    (off = expandtruth(img_boxes, off, ( V, H, 2)),
    size = expandtruth(img_boxes, sizeL, ( V, H, 2)),
    class = expandtruth(img_boxes, class, ( V, H, length(cat_ids))),
    heatmap = heatmap(img_boxes) |> expandim)

end



const imgdir = raw"D:\COCO\small"
xiter = mapobs(names) do img_name

    img = joinpath(imgdir, img_name) |> Images.load
    
    enc = ImagePreprocessing()
    encode(enc, Training(), FastVision.Image{2}(), img)
end


















# ANCHOR - end, code below is not used

# y = getobs(yiter, 1:4) |> batch

# y.mask[:,:,1,1] |> CairoMakie.heatmap
# y.heatmap[:,:,1,1] |> CairoMakie.heatmap
# y.size[:,:,1,1] |> CairoMakie.heatmap



x = getobs(xiter, 2)
CairoMakie.heatmap(x[:,:,1][v:v+vw, h:h+hw])
CairoMakie.heatmap(x[:,:,1])


y = getobs(yiter, 1:10) |> batch

CairoMakie.heatmap(y.heatmap[:,:,1,1])


v,h, vw, hw = img_boxes[1]


img = joinpath(imgdir, name) |> load


boxes = map(bbox_dict[name]) do b
    b.bbox
end

guidict = imshow(img)
for pred in boxes
    # draw!(img, Polygon(RectanglePoints(ImageDraw.Point(pred.x, pred.y), ImageDraw.Point(pred.x+pred.w, pred.y+pred.h))), colorant"yellow")
    annotate!(guidict, AnnotationBox(pred[2], pred[1], pred[2]+pred[4], pred[1]+pred[3], linewidth=2, color=colorant"yellow"))
end
guidict






