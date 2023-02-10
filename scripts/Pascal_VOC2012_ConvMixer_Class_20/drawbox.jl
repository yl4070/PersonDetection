
using Images
using ImageDraw
using ImageView

const IMSIZE = 256

function getimgencode(imgname)
    sz = IMSIZE
    if test
        imgdir = raw"D:\Datasets\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages"
    else
        imgdir = raw"D:\Datasets\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages"
    end
    color0 = colorant"black"
    img_file = joinpath(imgdir, imgname)
    
    # imgdir = raw"C:\Datasets\swimcar\training_data\training_data\images"
    if isfile(img_file)
        img = Images.load(img_file)
        w, h = size(img)
        w, h = w > h ? (sz, round(sz/w*h)) : (round(sz/h*w), sz)
        img = imresize(img, (Int(w), Int(h))) |> x -> PaddedView(color0, x, (sz, sz)) |> Array
        enc = ImagePreprocessing()
        encimg = encode(enc, Training(), FastVision.Image{2}(), img)
        imgx = Flux.unsqueeze(encimg, 4)

    # elseif imgname ∈ imgnames
    #     imgid = findfirst(imgnames .== imgname)
    #     imgx = getobs(xiter, imgid)
    #     imgx = Flux.unsqueeze(imgx, 4)        
    #     img = joinpath(imgdir, imgname ) |> Images.load
    #     img = imresize(img, (IMSIZE, IMSIZE))

    else
        error("Image not found")
    end

    img, imgx
end


function drawbox(model, imgname, xiter = xiter; threshold = .3, names = names)

    img, imgx = getimgencode(imgname)

    img = imresize(img, (IMSIZE, IMSIZE)) # NOTE - added

    yhat = model(imgx)

    idx = findall(el -> el > threshold, yhat[3][:,:,1,1])

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

    for (id, szv, szh, offv, offh) in zip(idx, sizevs, sizehs, offvs, offhs)
        
        v, h = id[1], id[2]

        x, y = (h + offh, v + offv)
        x, y = (4x-szh/2, 4y-szv/2) .|> round .|> Int

        szh, szv = (szh, szv) .|> round .|> Int

        draw!(img, Polygon(RectanglePoints(ImageDraw.Point(x, y), ImageDraw.Point(x+szh, y+szv))), colorant"yellow")
    end

    img

end

function drawboxView(img, preds::Vector)
    
    # colors = [colorant"yellow", colorant"blue", colorant"green"]
    colors = ColorScheme(distinguishable_colors(20, transform=protanopic))

    img = imresize(img, (IMSIZE, IMSIZE))
    # pred = preds[1]
    guidict = imshow(img)
    for pred in preds
        # draw!(img, Polygon(RectanglePoints(ImageDraw.Point(pred.x, pred.y), ImageDraw.Point(pred.x+pred.w, pred.y+pred.h))), colorant"yellow")

        annotate!(guidict, AnnotationBox(pred.x, pred.y, pred.x+pred.w, pred.y+pred.h, linewidth=2, color=colors[parse(Int64,pred.class)]))
        annotate!(guidict, AnnotationText(pred.x, pred.y+pred.w÷2, pred.class, color = colorant"red", fontsize = 8))
    end
    guidict
end

function drawboxview(model::Chain, imgname::String)

    img, imgx = getimgencode(imgname)
    preds = getprediction(model, imgx)
    preds = nonMaxSupression(preds, .3)

    drawboxView(img, preds)
end


function getdetection(model, imgname::AbstractString) # NOTE - deleted model's datatype
    _, imgx = getimgencode(imgname)

    preds = getprediction(model, imgx)
    # preds = nonMaxSupression(preds, .3)

    preds
end


# drawboxview(model, imgnames[1])

# drawbox(model, "175"; threshold = .5)
# drawbox(model, "001"; threshold = .5)


# CairoMakie.heatmap(yhat[4][:,:,1,1])
# CairoMakie.heatmap(getobs(yiter, 40).mask[:,:,1])



