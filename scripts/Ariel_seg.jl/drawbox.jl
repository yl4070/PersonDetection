
using Images
using ImageDraw
using ImageView

function getimgencode(imgname, names = names)

    if isfile(imgname)

        img = load(imgname)
        img = imresize(img, (256, 256))
        enc = ImagePreprocessing()
        encimg = encode(enc, Training(), FastVision.Image{2}(), img)
        imgx = Flux.unsqueeze(encimg, 4)
    elseif imgname ∈ names

        imgid = findfirst(names .== imgname)
        imgx = getobs(xiter, imgid)
        imgx = Flux.unsqueeze(imgx, 4)
        
        img = joinpath(imgdir, imgname * ".jpg") |> load
    else

        error("Image not found")
    end

    img, imgx
end


function drawbox(model, imgname, xiter = xiter; threshold = .5, names = names)

    img, imgx = getimgencode(imgname)

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

    guidict = imshow(img)
    for pred in preds
        # draw!(img, Polygon(RectanglePoints(ImageDraw.Point(pred.x, pred.y), ImageDraw.Point(pred.x+pred.w, pred.y+pred.h))), colorant"yellow")
        annotate!(guidict, AnnotationBox(pred.x, pred.y, pred.x+pred.w, pred.y+pred.h, linewidth=2, color=colorant"yellow"))
        annotate!(guidict, AnnotationText(pred.x, pred.y+pred.w÷2, string( round(pred.uncern[1]; digits=2) ), color = colorant"red", fontsize = 8))
    end
    guidict
end


function drawboxview(model::Chain, imgname::String)

    img, imgx = getimgencode(imgname)
    preds = getprediction(model, imgx)
    nonMaxSupression!(preds, .3)

    drawboxView(img, preds)
end



drawboxview(model, "002")


drawbox(model, "175"; threshold = .5)
drawbox(model, "001"; threshold = .5)


CairoMakie.heatmap(yhat[4][:,:,1,1])
CairoMakie.heatmap(getobs(yiter, 40).mask[:,:,1])



