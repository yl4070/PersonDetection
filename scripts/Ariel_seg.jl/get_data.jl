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
imgnames = keys(boxes) |> collect
persondir = raw"D:\training_set\personmasks"

yiter = mapobs(names) do name

    img_boxes = boxes[name]

    off = map(img_boxes) do box
        cal_offset(box...)
    end

    sizeL = map(img_boxes) do box
        box[3:4]
    end

    img = joinpath(persondir, name * ".png") |> Images.load
    img = imresize(img, (V, H))
    mask = channelview(Gray.(img)) |> Array |> f32 
    ol = 1 .- mask

    # return named-tuple
    (off = expandtruth(img_boxes, off, ( V, H, 2)),
    size = expandtruth(img_boxes, sizeL, ( V, H, 2)),
    heatmap = heatmap(img_boxes) |> expandim,
    # points = centerize.(img_boxes),
    mask = cat(mask, ol; dims = 3))
end


const imgdir = raw"D:\training_set\small"
xiter = mapobs(imgnames) do name

    img_name = name * ".jpg"
    img = joinpath(imgdir, img_name) |> Images.load
    
    img = imresize(img, (256, 256))

    enc = ImagePreprocessing()
    encode(enc, Training(), FastVision.Image{2}(), img)
end




# ANCHOR - end, code below is not used

# y = getobs(yiter, 1:4) |> batch

# y.mask[:,:,1,1] |> CairoMakie.heatmap
# y.heatmap[:,:,1,1] |> CairoMakie.heatmap
# y.size[:,:,1,1] |> CairoMakie.heatmap





 