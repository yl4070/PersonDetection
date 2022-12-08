using MLUtils
using Images
using ImageDraw
using ProgressMeter


visdrone = raw"C:\VisDrone2019-VID-train"
person_dir = raw"C:\VisualProject\imgs"

anno, seqs = readdir(visdrone)

anno_dir = joinpath(visdrone, "annotations")
annotations = readdir(anno_dir)

get_path(file) = joinpath(anno_dir, file)

files = mapobs(get_path, annotations)
area_rec = Dict()

for f in eachobs(files)
    open(f) do ff
        area = 0
        for l in eachline(ff)
            lst = parse.(Int, split(l, ","))
            w, h = lst[5:6]
            if lst[1] == 1

                if lst[end-2] == 0
                    area += w*h
                end
            end
        end
        seq_name = split(f, "\\")[end]
        area_rec[seq_name] = area
    end
end

sort!(annotations; by = x->area_rec[x])

no_ignore = filter(annotations) do x
    area_rec[x] == 0
end


function loadimg(f, seq)

    seqpath = joinpath(seq_dir, seq)
    anno_path = joinpath(anno_dir, seq * ".txt")
    path = joinpath(visdrone, "sequences", seqpath, f)

    lines = open(anno_path) do annos
        readlines(annos)
    end
    lines = filter(lines) do l
        parse(Int, split(l, ",")[1]) == parse(Int, split(f,".")[1]) && parse(Int, split(l, ",")[end-2]) ∈ [1, 2]
    end
    img_name = "$(seq)_$(split(f, '.')[1])"
    bbox = map(lines) do l
        bstr = split(l, ",")[3:6]
        img_name * "," * join(bstr, ",")
    end
    load(path), bbox, img_name
end

seq_dir = joinpath(visdrone, "sequences")

global im, ann_lines, name

ann_lst = []

person_dir = raw"C:\VisualProject\imgs"

function loadperson(f)
    path = joinpath(person_dir, f)
    load(path)
end

persons = mapobs(loadperson, readdir(person_dir))

# ANCHOR add person
function attachPerson!(img::AbstractArray, coords::Tuple{Vararg{T} where T <: Integer}, person::AbstractArray) 

   x, y = coords
   w, h = size(person) .÷ 2

   for i in x-w:x+w-1, j in y-h:y+h-1
      a, b = i-x+w+1, j-y+h+1
      img[i, j] = person[a, b] == color0 ? img[i,j] : person[a,b]
   end  
   img
end

function tinify(person, sz)

    tsz = rand([20, 30], 1)[1]
    imresize(person, sz .÷ tsz) 
end
function randpos(sz)
    margin = sz .÷ 40
    ul = sz .- margin .- 1
    ll = 1 .+ margin .+ 1
    (rand(ll[1]:ul[1], 1), rand(ll[2]:ul[2], 1)) .|> first
end


all_bboxs = []

shuffleobs(no_ignore)

outpath = raw"C:\VisualProject\Aug_drone"

@showprogress for seq in shuffleobs(no_ignore)
    seq = split(seq, ".")[1]

    seqpath = joinpath(seq_dir, seq)
    loadimg(f) = loadimg(f, seq)
    imgs = mapobs(loadimg, readdir(seqpath))

    for (img, bbox, im_name) in eachobs(imgs, shuffle = true)

        if length(bbox) < 40

            n_add = 40 - length(bbox)
            pers = randobs(persons, n_add)
            for per in pers
                
                person = tinify(per, size(img))
                psz = size(person)
                pos = randpos(size(img))
                box = [(pos .- (psz .÷ 2))..., psz...]
                y1, x1, h, w = box

                attachPerson!(img, pos, person)

                push!(bbox, join((im_name, x1, y1, w, h), ","))
            end
        end
        all_bboxs = vcat(all_bboxs, bbox)

        save(joinpath(outpath, im_name * ".png"), img)
    end
end

(img, bbox, im_name) = getobs(imgs, 1)
draw!(img, Polygon(RectanglePoints(Point(x1, y1), Point(x1+w, y1+h))), RGB{N0f8}(1))

all_bboxs

bbox = lab_dict["uav0000288_00001_v_0000233"]

for l in bbox
    l = split(l, ",")
    l = parse.(Int, l)
    x1, y1 = l[1:2]
    w, h = l[3:4]
    centers = calc_center([x1, y1, w, h])
    draw!(testx, Ellipse(CirclePointRadius(centers..., 3)))
    draw!(testx, Polygon(RectanglePoints(ImageDraw.Point(x1, y1), ImageDraw.Point(x1+w, y1+h))), RGB{N0f8}(1))
end
testx = im

testx
bbox = lab_dict[img]
testx

using BSON: @save

@save "all_bboxs.bson" all_bboxs

testx = load(raw"C:\VisualProject\Aug_drone\uav0000288_00001_v_0000300.png")

txt = "uav0000288_00001_v_0000300"

bbox = filter(all_bboxs) do x
    split(x, ",")[1] == txt
end

using ImageDraw


for l in bbox
    x1, y1 = l[1]
    x2, y2 = l[2]
    # draw!(testx, Ellipse(CirclePointRadius(centers..., 3)))
    draw!(testx, ImageDraw.Polygon(RectanglePoints(ImageDraw.Point(x1, y1), ImageDraw.Point(x1+w, y1+h))), RGB{N0f8}(1))
end
testx = im
testx
























