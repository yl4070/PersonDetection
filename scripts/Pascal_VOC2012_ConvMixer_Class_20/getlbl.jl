using JSON3
using BSON: @save
using ProgressMeter


function get_info(pascal_dir, pascal_lbl_tr)
    # const pascal_dir = raw"D:\pascal_2007\train"
    # const pascal_lbl_tr = raw"D:\pascal_2007\train.json"

    json_string = read(pascal_lbl_tr, String)
    raw_lbl = JSON3.read(json_string)
    annotations = raw_lbl[:annotations] 
    img_info = raw_lbl[:images]

    img_names = readdir(pascal_dir) |> Set
    info_dict = Dict()

    for img in img_info        
        info_dict[img.id] = (height = img.height, width = img.width, file_name = img.file_name)
    end

    annotations, img_names, info_dict
end

function check_bound(x; sz)

    if x > sz
        sz
    elseif x < 1
        1
    else
        x
    end
end

function intify(x)
    round(x) |> Int
end


function getbbox(annotations, img_names, info_dict; sz = 256)
    
    bbox_dict = Dict()

    @showprogress for lbl in annotations

        h, v, hw, vw = lbl[:bbox]

        img_id = lbl[:image_id]

        if !haskey(info_dict, img_id) 
            continue
        end

        img_info = info_dict[img_id]
        iv, ih = img_info.height, img_info.width

        if img_info.file_name ∉ img_names
            continue
        end

        # NOTE - Resize to input size images
        # v, vw = @. (v, vw) / iv * sz
        # h, hw = @. (h, hw) / ih * sz
        
        # vw, hw = @. (vw, hw) |> ceil |> Int64
        # v, h = @. (v, h) |> round |> Int64

        # NOTE - keep aspect ratio
        mx =  iv > ih ? sz / iv : sz/ih # mx => multiplier

        v = intify(v * mx)
        h = intify(h * mx)

        vw, hw = intify.((vw * mx, hw * mx))
    
        v, h = check_bound.([v, h]; sz = sz)   

        box = (cat_id = lbl[:category_id], bbox = [v, h, vw, hw])
        bbox_dict[img_info.file_name] = push!(get(bbox_dict, img_info.file_name, []), box)
    end

    bbox_dict
end

# bbox_dict = getbbox(annotations, img_names, info_dict)

# @save "bbox_dict_pascal.bson" bbox_dict
# @save "cat_id_pascal.bson" cat_ids

# cat_ids = Set()

# for anno in annotations
#     push!(cat_ids, anno.category_id)
# end

# cat_ids = cat_ids |> collect





