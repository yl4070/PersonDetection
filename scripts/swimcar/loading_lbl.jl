
using EzXML


function check_bounds(v)
    if v <= 0
        1
    elseif v >= 256
        256
    else
        v
    end
end

parstint(x) = parse(Int, x)
Base.round(x::String) = parse(Float64, x) |> round
Base.ceil(x::String) = parse(Float64, x) |> ceil

const swim_lbl_tr = raw"D:\swimcar\training_data\labels"
bboxes = Dict()

for xml_file in readdir(swim_lbl_tr)

    doc = readxml(joinpath(swim_lbl_tr, xml_file))
    primates = root(doc)

    
    imgname = findfirst("//filename", primates).content

    bboxes[imgname] = []

    imgh = findfirst("//width", primates).content |> parstint
    imgv = findfirst("//height", primates).content |> parstint

    for box = findall("//object", primates)

        h = findfirst(".//xmin", box).content |> round |> Int
        v = findfirst(".//ymin", box).content |> round |> Int

        hw = (findfirst(".//xmax", box).content |> ceil |> Int) - h
        vw = (findfirst(".//ymax", box).content |> ceil |> Int) - v

        class = findfirst(".//name", box).content |> parstint

        # NOTE - Resize to 256x256 images
        v, vw = @. (v, vw) / imgv * 256 
        h, hw = @. (h, hw) / imgh * 256
        
        vw, hw = @. (vw, hw) |> ceil |> Int64
        v, h = @. (v, h) |> round |> Int64

        v, h = check_bounds.([v, h])
        push!(bboxes[imgname], (bbox = [v, h, vw, hw], class = class))
    end

end

# ANCHOR End here, code below not used
const swim_img_dir = raw"D:\swimcar\training_data\images"
# img = raw"D:\training_set\images\475.jpg"

function drawboxView(imgname)
    colors = [colorant"yellow", colorant"blue"]
    imgpath = joinpath(swim_img_dir, imgname)
    img = Images.load(imgpath)
    img = imresize(img, (256, 256))

    guidict = imshow(img)
    for box in bboxes[imgname]
        # draw!(img, Polygon(RectanglePoints(ImageDraw.Point(pred.x, pred.y), ImageDraw.Point(pred.x+pred.w, pred.y+pred.h))), colorant"yellow")
        annotate!(guidict, AnnotationBox(box.bbox[2], box.bbox[1], box.bbox[2]+box.bbox[4], box.bbox[1]+box.bbox[3], linewidth=2, color=colors[box.class]))
    end
    guidict
end

drawboxView("000000227.jpg")



# img = Images.load(img)
# # boxes["101"]

# # (4000, 6000) .÷ 20


# v, h, vw, hw = boxes["475"][3]
# # img[v:v+vw, h:h+hw]


# # img = imresize(img, (200, 200))

# # boxes["475"]


# # img = joinpath(persondir, "101.png")


# (v, h, vw, hw) 

# ceil(1.2)
# v, vw = @. v, vw / 4000 * 256

# h, hw = @. h, hw / 6000 * 256



# img[]
























