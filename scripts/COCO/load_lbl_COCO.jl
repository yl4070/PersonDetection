
using JSON3
using BSON: @save

struct ImgObject
    category_id::String
    bbox::Vector{Float32}
    image_id::Int
end

coco_dir = "D:\train2017\train2017"

lbl_path = raw"D:\annotations\instances_train2017.json"

json_string = read(lbl_path, String)
raw_lbl = JSON3.read(json_string)
annotations = raw_lbl[:annotations] 
img_info = raw_lbl[:image]


img_names = readdir(coco_small_dir)

img_dict = Dict()
for img in img_names

    img_id = parse(Int, split(img, ".")[1])
    img_dict[img_id] = img
end


function getbbox(annotations, img_dict)
    
    bbox_dict = Dict()
    coco_dir = raw"D:\train2017\train2017"

    @showprogress for lbl in annotations

        h, v, hw, vw = lbl[:bbox]

        name = lbl[:image_id]

        if !haskey(img_dict, name)
            continue
        end

        img = joinpath(coco_dir, img_dict[name]) |> load

        iv, ih = size(img)

        # NOTE - Resize to 256x256 images
        v, vw = @. (v, vw) / iv * 256
        h, hw = @. (h, hw) / ih * 256
        
        vw, hw = @. (vw, hw) |> ceil |> Int64
        v, h = @. (v, h) |> round |> Int64

        v, h = check_bounds.([v, h])   

        box = (cat_id = lbl[:category_id], bbox = [v, h, vw, hw])
        bbox_dict[img_dict[name]] = push!(get(bbox_dict, img_dict[name], []), box)
    end

    bbox_dict
end

bbox_dict = getbbox(annotations, img_dict)

@save "bbox_dict_coco.bson" bbox_dict


h, v, hw, vw = (h, v, hw, vw) .|> round .|> Int

img = imresize(img, (256, 256))

lbl = annotations[10]
img[v:v+vw, h:h+hw]



cats = map(annotations) do anno 
    anno[:category_id]
end
cm = countmap(cats)
cat_ids = keys(cm) |> collect

CairoMakie.hist(cats)

cnts = counts(cats)
maximum(cnts)
minimum(cnts)


ids = []
for id in keys(cm)

    cnts = cm[id]

    if cnts > 1000 && cnts < 2000
        push!(ids, id)
    end
end


ids



