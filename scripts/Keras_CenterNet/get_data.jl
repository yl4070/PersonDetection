

# const V, H = 128, 128 # REVIEW - check output size later.

Base.:

function cal_offset(v, h, vw, hw, R = 4)
    p = [v + vw÷2, h + hw÷2]
        
    p̃ = p .÷ R
    @. p / R - p̃
end

# cal_offset(boxes["006"][1]...)
function centerize(box, R = 4)
    v, h, vw, hw = box
    p = [v + vw÷2, h + hw÷2]

    check_bound.(p .÷ R; sz = 64)
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

    Y = zeros(V, H, 20)

    for (box, id) in boxes

        Y′ = zeros(V, H, 20)
        p̃ = centerize(box)
        
        # p̃i_range = min(p̃[1]-5, 1):max(p̃[1]+5, V)
        # p̃j_range = min(p̃[2]-5, 1):max(p̃[2]+5, H)
        
        p̃i_range = max(p̃[1]-5, 1):min(p̃[1]+5, V)
        p̃j_range = max(p̃[2]-5, 1):min(p̃[2]+5, H)

        for i in p̃i_range, j in p̃j_range
            Y′[i, j, id] = Yxyc(i, j, p̃[1], p̃[2])
        end

        Y = max.(Y, Y′)
    end

    Y
end

expandim(x, d = 3) = Flux.unsqueeze(x, d)

# ANCHOR - get label data

function get_data(bbox_dict, imgdir; sz = 256)
        
    color0 = colorant"black"

    imgnames = keys(bbox_dict) |> collect
    
    yiter = mapobs(imgnames) do iname

        img_boxes = map(bbox_dict[iname]) do b
            b.bbox, b.cat_id
        end

        pure_boxes = map(img_boxes) do b
            b[1]
        end

        off = map(img_boxes) do b
            cal_offset(b[1]...)
        end

        sizeL = map(img_boxes) do b
            b[1][3:4]
        end

        (off = expandtruth(pure_boxes, off, (V, H, 2)),
        size = expandtruth(pure_boxes, sizeL, (V, H, 2)),
        # class = expandtruth(img_boxes, class, ( V, H, length(cat_ids))),
        heatmap = heatmap(img_boxes))
    end

    xiter = mapobs(imgnames) do img_name

        img = joinpath(imgdir, img_name) |> Images.load
        
        # img = imresize(img, (sz, sz))

        w, h = size(img)
        w, h = w > h ? (sz, round(sz/w*h)) : (round(sz/h*w), sz)
        img = imresize(img, (Int(w), Int(h))) |> x -> PaddedView(color0, x, (sz, sz)) |> Array

        enc = ImagePreprocessing()
        encode(enc, Training(), FastVision.Image{2}(), img)
    end

    xiter, yiter
end
# bbox_dict["009709.jpg"]

# x = getobs(xiter, 2)
# yy = getobs(yiter, 2)

# using CairoMakie

# CairoMakie.heatmap(x[:,:,1])
# CairoMakie.heatmap(yy.heatmap[:,:,5])

