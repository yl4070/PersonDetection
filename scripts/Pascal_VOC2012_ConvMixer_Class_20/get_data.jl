# const V, H = 128, 128 # REVIEW - check output size later.

Base.:÷(a, b::AbstractFloat) = round(a / b) |> Int

function cal_offset(v, h, vw, hw; R = RATIO)
    p = [v + vw÷2, h + hw÷2]
        
    p̃ = p .÷ R
    @. p / R - p̃
end

# cal_offset(boxes["006"][1]...)
function centerize(box; R = RATIO, imsize = imsize)
    v, h, vw, hw = box
    p = [v + vw÷2, h + hw÷2]

    check_bound.(p .÷ R; sz = imsize ÷ R)
end

function expandtruth(boxes, ys, dim; R, imsize)

    p̃s = centerize.(boxes; R, imsize)

    L = zeros(dim)    
    for (p̃, y) in zip(p̃s, ys)
        
        L[p̃..., :] = y
    end
    L
end

# size = 256x256
Yxyc(v, h, pv, ph) = exp(-((v-pv)^2 + (h-ph)^2)/2)
function heatmap(boxes; V, H, R, imsize)

    Y = zeros(V, H, 20)

    for (box, id) in boxes

        Y′ = zeros(V, H, 20)
        p̃ = centerize(box; R, imsize)
        
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

function get_ydata(imglist, bbox_dict, imsize, V)

    # imgnames = keys(bbox_dict) |> collect

    H = V    
    ratio = imsize / H

    yiter = mapobs(imglist) do iname

        classes = split("person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, diningtable, pottedplant, sofa, tvmonitor", ", ")
            

        img_boxes = map(bbox_dict[iname]) do b
            b.bbox, findfirst(c -> c == b.class, classes)
        end

        pure_boxes = map(img_boxes) do b
            b[1]
        end

        off = map(img_boxes) do b
            cal_offset(b[1]...; R = ratio)
        end

        sizeL = map(img_boxes) do b
            b[1][3:4]
        end

        
        (off = expandtruth(pure_boxes, off, (V, H, 2); R = ratio, imsize = imsize),
        size = expandtruth(pure_boxes, sizeL, (V, H, 2); R = ratio, imsize = imsize),
        heatmap = heatmap(img_boxes; H = H, V = V, R = ratio, imsize = imsize))
    end


    yiter
end

function get_xdata(imglist, imgdir; sz = 256)
        
    color0 = colorant"black"

    # imgnames = keys(bbox_dict) |> collect
    
    xiter = mapobs(imglist) do img_name

        img = joinpath(imgdir, img_name) |> Images.load
        
        w, h = size(img)
        w, h = w > h ? (sz, round(sz/w*h)) : (round(sz/h*w), sz)
        img = imresize(img, (Int(w), Int(h))) |> x -> PaddedView(color0, x, (sz, sz)) |> Array

        enc = ImagePreprocessing()
        encode(enc, Training(), FastVision.Image{2}(), img)
    end

    xiter
end


struct TrData
    xiter
    yiter1
    yiter2
    yiter3
end



Base.getindex(d::TrData, i) = getindex(d.xiter, i), getindex(d.yiter1, i), getindex(d.yiter2, i), getindex(d.yiter3, i)
Base.length(d::TrData) = length(d.xiter)


# y
# x, y1, y2, y3 = getobs(trd, 1)




# tot_loss(y, y1, y2, y3, pos_idx, flt_idx)