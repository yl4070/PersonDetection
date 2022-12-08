using GeometricalPredicates
using ProgressMeter
using FastAI, FastVision

cropedir = joinpath(projectdir(), "cropedir")



selected = ["uav0000076_00720", "uav0000288_00001"]

selected_names = filter(collect(keys(lab_dict))) do key
    occursin(selected[1], key) || occursin(selected[1], key)
end

imgiter = mapobs(loadimg, selected_names)


img = getobs(imgiter, 1)

function getbbox(lab)

    lab = lab_dict[lab]
    lab = map(lab) do str_coords
        x, y, w, h = parse.(Int, split(str_coords, ","))
    end

    map(lab) do l
        x, y, w, h = l
        (y,x), (y+h, x+w)
    end
end

bboxiter = mapobs(getbbox, selected_names)

bbox = getobs(bboxiter, 1)[2]


function inbox(centers::Tuple, bbox::Tuple)

    for (y, x) in bbox
        p = GeometricalPredicates.Point(y, x)
        ll = GeometricalPredicates.Point(centers[1]+128, centers[2]-127)
        lr = GeometricalPredicates.Point(centers[1]+128, centers[2]+128)
        ur = GeometricalPredicates.Point(centers[1]-127, centers[2]+128)
        ul = GeometricalPredicates.Point(centers[1]-127, centers[2]-127)
        poly = GeometricalPredicates.Polygon(ll, lr, ur, ul)
        if ! inpolygon(poly, p)
            return false
        end
    end
    true
end


getobs(bboxiter, 1)

bbox_dict = Dict{String, Any}()
imgn=1
@showprogress for (img, bboxs) in zip(eachobs(imgiter), eachobs(bboxiter))

    vert, hori = size(img)
    margin = 256 ÷ 2 + 1

    i = 0

    for _ in 1:100

        centerx = rand(margin:hori-margin, 1)[1]
        centery = rand(margin:vert-margin, 1)[1]

        bb = filter(bboxs) do bbox
            inbox((centery, centerx), bbox)
        end

        bb = map(bb) do box
            ((y1, x1), (y2, x2)) = box
            
            y1 = y1 -centery+127
            y2 = y2 -centery+127

            x1 = x1 -centerx+127
            x2 = x2 -centerx+127
            (y1, x1), (y2, x2)
        end
        if length(bb) >= 5
            savename = "img$(imgn)_$(i)_$(centerx)_$centery.png"
            Images.save(joinpath(cropedir, savename), img[centery-127:centery+128, centerx-127:centerx+128])
            bbox_dict[savename] = vcat(get(bbox_dict, savename, []), bb)
            i+=1
        end
        if i > 5
            break
        end
    end
    imgn += 1
end


using BSON: @save, @load
@save "lab_dict.bson" lab_dict
@load "lab_dict.bson" lab_dict

@save "bbox_dict.bson" bbox_dict
@load "bbox_dict.bson" bbox_dict

@load "uNet_m.bson" cpu_m

# ANCHOR Prepare BBox

function getcropped(img)
    path = joinpath(cropedir, img)
    Images.load(path)
end
function getencoded(img)
    path = joinpath(cropedir, img)
    im = Images.load(path)

    enc = ImagePreprocessing()
    encode(enc, Training(), FastVision.Image{2}(), im)
end


encx = encode(enc, Training(), FastVision.Image{2}(), im)
im = testx


imgnames = collect(keys(bbox_dict))

# ANCHOR input data
croppediter = mapobs(getcropped, imgnames)
encodediter = mapobs(getencoded, imgnames)

img = imgnames

heatmap(newx[:,:,1] |> cpu)


newx = getobs(encodediter, 2) 
yy = m(Flux.unsqueeze(newx, 4)) |> cpu

heatmap(yy[:,:,1,1])
heatmap(newx[:,:,1] |> cpu)
Gray.(yy[:,:, 1,1])


crop = getobs(croppediter, 2) 
box = bbox_dict[imgnames[2]][6]

crop[box[1][1]:box[2][1], box[1][2]:box[2][2]]


# ANCHOR Lable
lbliter = mapobs(getlabel, imgnames)

lab = imgnames[2]
bbox_dict[imgnames[1]]


y = getobs(lbliter, 2)

getobs(croppediter, 2)


# Training BBox

backbone = Metalhead.ResNet(18).layers[1:end-1]
model = Chain(
    Conv((1,1), 2 => 3),
    backbone[1][1:end-6],
    Conv((1,1), 64 => 5),
    BatchNorm(5, relu)
)

m = gpu(model)

ŷ = m(newx)



dl = DataLoader((encodediter, lbliter); batchsize = 8, shuffle = true, collate = true)

encm = gpu(cpu_m)
opt = Adam()
ps = Flux.params(m)

for e in 1:20
    @info "epoch" e
    i = 0
    for (x, y) in dl

        local loss

        x, y = x |> gpu, y |> gpu

        x = encm(x)

        gs = gradient(ps) do 
            ŷ = m(x)
            # loss = batch_loss(ŷ, y)
            loss = Flux.mse(ŷ, y)
            loss
        end
        
        Flux.update!(opt, ps, gs)

        if i % 10 == 0
            @show loss
        end

        i += 1
    end
end

ŷ = rand(Float32, 64, 64, 5, 8) |> gpu
y = rand(Float32, 64, 64, 5, 8) |> gpu

batch_loss(Ŷ, Y)

x = rand(Float32, 256, 256, 3, 8) |> gpu

m(x)

y = gpu(y)

ŷ[:,:,2,1] .* y[:,:,6, 1]

test = raw"C:\Users\yl4070\Pictures\imag.png"

testx = Images.load(test)[1:395, 1:395]

testx = imresize(testx, 256, 256)


heatmap(x)
x = getcropped("img89_1_311_920.png")
testx = getencoded("img89_1_311_920.png")

testx = getobs(croppediter, 25)
testx = getobs(encodediter, 25)
heatmap(testx[:,:,1])

heatmap(testx)

testx = encx
encx = encm(Flux.unsqueeze(gpu(testx), 5))
yhat = m(encx)

heatmap(cpu(encx[:,:,1,1]))

heatmap(yhat[:,:,1,1])

getobs(croppediter, 12)

img = Gray.(cpu(2encx[:,:,2,1]))

label = label_components(img)
bbox = component_boxes(label)[3:end]

box = bbox[1]

img[box[1][1]:box[2][1], box[1][2]:box[2][2]] 

closing(img)

img = erode(img)


using ImageSegmentation


img2 = felzenszwalb(img, 10000)

maph = labels_map(img2)
 
heatmap(label)

bbox = filter(bbox) do box

    w, h = box[2][1] - box[1][1], box[2][2]-box[1][2]
    w*h > 10
end


box = bbox[7]


bbox = component_boxes(labels_map(unseeded_region_growing(img, .7)))
clearborder(img, 30)




testx = img





