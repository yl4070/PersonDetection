using Images
using MLUtils
using FastAI, FastVision

const V, H = 64, 64

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
imgnames = keys(bboxes) |> collect

yiter = mapobs(imgnames) do name

    img_boxes = map(bboxes[name]) do b
        b.bbox
    end

    off = map(img_boxes) do box
        cal_offset(box...)
    end

    sizeL = map(img_boxes) do box
        box[3:4]
    end

    # REVIEW - all classes for now, 20
    class = map(bboxes[name]) do obj
        obj.class |> id -> Flux.onehot(id, [1,2])
    end

    # return named-tuple
    (off = expandtruth(img_boxes, off, ( V, H, 2)),
    size = expandtruth(img_boxes, sizeL, ( V, H, 2)),
    class = expandtruth(img_boxes, class, ( V, H, 2)),
    heatmap = heatmap(img_boxes) |> expandim)

end



# y = getobs(yiter, 1)

# CairoMakie.heatmap(y.heatmap[:,:,1,1])


# img_name = imgnames[1]
xiter = mapobs(imgnames) do img_name

    img = joinpath(swim_img_dir, img_name) |> Images.load
    
    img = imresize(img, (256, 256))
    enc = ImagePreprocessing()
    encode(enc, Training(), FastVision.Image{2}(), img)
end

# x = getobs(xiter, 1:10) |> batch
# y = getobs(yiter, 1:10) |> batch

# CairoMakie.heatmap(y.heatmap[:,:,1, 1])

# CairoMakie.heatmap(x[:,:,1, 1])

# yhat = model(x)

# CairoMakie.heatmap(yhat[3][:,:,1, 1])




















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






