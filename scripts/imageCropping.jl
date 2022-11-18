using Images
using FileIO
using ProgressMeter

folder = raw"C:\VisDrone2019-VID-train"
annotation, sequences = readdir(folder)

make_name(nlist) = reduce(string.(nlist)) do x, y
    x * "_" * y
end

vidpath = joinpath(folder, sequences) |> readdir

seq = vidpath[1]

imgpath = joinpath(folder, sequences, seq)

imgs = readdir(imgpath)

img1 = joinpath(imgpath, imgs[1]) |> load

img1

annopath = joinpath(folder, annotation) |> readdir

img = readdir(imgpath)[1]

for seq in vidpath

    imgpath = joinpath(folder, sequences, seq)

    for img in readdir(imgpath)
        
        img_id = parse(Int, split(img, ".")[1])

        img1 = load(joinpath(imgpath,img))

        annopath = joinpath(folder, annotation, seq) * ".txt"
        f = open(annopath)

        while !eof(f)

            line = split(readline(f), ",") .|> x -> parse(Int, x)

            id = line[1]
            x1, y1, w, h = line[3:6] .+ [1,1,0,0]
        
            if id == img_id && line[8] ∈ [1,2] && line[9] == 0 && line[10] != 2

                imgf = Float32.(channelview(img1))
                v = var(imgf) 
                fname = string(1000v) * "_" * string(img_id) * "_" * make_name(line[3:6]) * ".png"

                save(joinpath(projectdir(), "imgs", seq, fname), img1[y1:y1+h-1, x1:x1+w-1])
            end
        end
        
        close(f)
    end
end

f = open(joinpath(folder, annotation, vidpath[1]) * ".txt") 

close(f)





# testing variance

imgs = readdir("imgs")


img92 = filter(imgs) do img
    id = split(img, "_")[1]
    parse(Int, id) == 92
end

img = load(joinpath("imgs", img92[1]))

testp = load(raw"C:\VisualProject\PersonCropping\imgs\92_518_375_38_45.png")

imgf = Float32.(channelview(img))
imgf2 = Float32.(channelview(testp))

using Statistics
 
var(imgf)
var(imgf2)

imgs_path = joinpath(projectdir(), "imgs")
var_img = joinpath(projectdir(), "var_img")

img  =  readdir(imgs_path)[1]
for img in readdir(imgs_path)

    im = load(joinpath(imgs_path, img))
    imgf = Float32.(channelview(im))

    v = var(imgf) 

    # img_name =  string(v) * "_" * split(img, ".")[1] * ".png"
    img_name =  string(1000v) * ".png"

    save(joinpath(var_img, img_name), im)
end







