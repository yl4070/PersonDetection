using Flux
using Metalhead
using MLUtils
using BSON: @load
using CairoMakie
using Images

# ANCHOR Loading labels
@load "all_bboxs.bson" all_bboxs
rawlabel = all_bboxs
lab = rawlabel[1]

lab_dict = Dict{String, Vector{String}}()

map(rawlabel) do lab
    name, coords... = split(lab, ",")
    lab_dict[name] = push!(get(lab_dict, name, String[]), join(coords, ","))
end

lab = collect(values(lab_dict))[1]

function calc_center(bbox)
    x1, y1, w, h = bbox
    x1 + w ÷ 2, y1 + h ÷ 2    
end

σ² = 1
Yxyc(x,y, px, py) = exp(-((x-px)^2 + (y-py)^2)/2σ²)
# Object detection as points:
# required info: px, py, size

vert, hori = 256, 256
R = 4
function getlabel(lab)

    # lab = map(lab) do str_coords
    #     x, y, w, h = parse.(Int, split(str_coords, ","))
    # end

    lab = map(bbox_dict[lab]) do coords

        y, x = coords[1]
        h, w = coords[2] .- coords[1]
        x, y, w, h
    end

    Y = zeros(Float32, vert ÷ R, hori÷ R)
    Y′ = zeros(Float32, vert ÷ R, hori ÷ R)
    O = zeros(Float32, vert ÷ R, hori ÷ R, 2)
    S = zeros(Float32, vert ÷ R, hori ÷ R, 2)
    M = zeros(Float32, vert ÷ R, hori ÷ R)

    ax, ay = axes(Y)

    for l in lab

        w, h = l[3:4]
        centers = calc_center(l)
        px, py = centers .÷ R

        for i in ax, j in ay
            Y′[i,j] = Yxyc(i, j, py, px)
        end

        offsets = (8px, 8py) .- centers
        O[py, px, 1] = offsets[1]
        O[py, px, 2] = offsets[2]

        M[py, px] = 0
        

        S[py, px, 1] = w
        S[py, px, 2] = h
    
        Y = max.(Y, Y′)
    end
    cat(Y, O, S; dims = 3)
end

# ANCHOR Loss function
σ² = 1
Yxyc(x,y, px, py) = exp(-((x-px)^2 + (y-py)^2)/2σ²)
# Ĥ is the 2d output matrix
function Lk(Ĥ, H)
    
    loss = 0
    for x in axes(H, 1), y in axes(H, 2)

        Ŷ = σ.(Ĥ[x, y])
        Y = H[x, y]
        loss += Y == 1 ? (1-Ŷ)^2*log(Ŷ) : (1-Y)^4 * Ŷ^2 * log(1-Ŷ)
    end

    -loss / 10
end

function Loff(Ô, O, M)

    # loss = 0
    # for i in eachindex(O)
    #     if O[i] != 0
    #         loss += abs(O[i] - Ô[i])
    #     end
    # end
    # loss / 5

    Flux.Losses.mse(Ô .* M, O)
end

function Lsz(Ŝ, S, M)

    # loss = 0
    # for i in eachindex(S)

    #     if S[i] != 0
    #         loss += abs(S[i] - Ŝ[i])
    #     end
    # end

    Flux.Losses.mse(Ŝ .* M, S)
end

tot_loss(Ŷ, Y) = Flux.binary_focal_loss(Ŷ[:,:,1], Y[:,:,1]) + .1Lsz(Ŷ[:,:,4:5], Y[:,:,4:5], Y[:,:,6]) + Loff(Ŷ[:,:,2:3], Y[:,:,2:3], Y[:,:,6])

function batch_loss(Ŷ, Y)

    tot = 0
    for i in axes(Y, 4)
        tot += tot_loss(Ŷ[:,:,:, i], Y[:,:,:,i])
    end
    tot
end



lab_dict

imgs_name = filter(collect(keys(lab_dict))) do x
    occursin("uav0000072_04488", x)
end


loadimg(img) = Images.load(joinpath(raw"C:\VisualProject\Aug_drone", img * ".png"))

im = loadimg(imgs_name[1])

lbl = getlabel(lab_dict[img])

heatmap(lbl[:, end:-1:1,1])

getimglbl(img) = getlabel(lab_dict[img])

labiter = mapobs(getimglbl, imgs_name)

Y = getobs(labiter, 1)

function getX(img)
   im = loadimg(img) 

    im = im[750:1000, 540:800] 
    im = imresize(im, (256, 256))
    enc = ImagePreprocessing()

    encode(enc, Training(), FastVision.Image{2}(), im)
#    permutedims(channelview(im), (2,3,1)) .|> Float32
end

inputiter = mapobs(getX, imgs_name)

dl = DataLoader((data = inputiter, label = labiter); collate = true, batchsize = 8)

im = loadimg(imgs_name[1])

testx = getobs(inputiter, 1) 

using BSON: @load
@load "uNet_m.bson" cpu_m


testx = getobs(inputiter, 1)


anotherx = getX("uav0000072_05448_v_0000211")

m = gpu(cpu_m)

heatmap( testx[:, :,1] )


yhat = m(Flux.unsqueeze(testx, 4) |> gpu) |> cpu

heatmap(testx[:,:, 1])
heatmap(yhat[:,:, 1 , 1])


Gray.(yhat[:,:, 1, 1])

yhat[:,:, 1]

# ANCHOR Prediction
ax = axes(Y)

bbox = []

for i in ax[1], j in ax[2]

    if Y[i, j, 1] > .5
        x1 = 1
    end
end

# Generate heatmap for the augmented images
X = getX("uav0000288_00001_v_0000233")

heatmap(X[:,:, 1])


testx = Images.load(raw"C:\VisualProject\Aug_drone\uav0000288_00001_v_0000233.png")


im = imresize(testx[500:1000, 200:700], (256, 256))
im = imresize(im, (256, 256))
anotherx = permutedims(channelview(im), (2,3,1)) .|> Float32

anotherx = gpu(anotherx)



sz = size(testx) .÷ 4

imresize(testx, sz)


im = Images.load(raw"C:\Users\yl4070\Pictures\image.png")


softmax( yhat[1,1,:,1])


yy = zeros(Float32, 256, 256)

ax = axes(yy)
for i in ax[1], j in ax[2]
    yy[i,j] = softmax(yhat[i,j,:,1])[1]
end

heatmap(yy)


newx = Conv((3,3), 2 => 3, relu; pad = SamePad())(yhat) |> gpu

model = model |> gpu

model(newx)






