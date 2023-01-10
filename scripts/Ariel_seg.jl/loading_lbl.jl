
using Images
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



xml_file = raw"D:\training_set\gt\bounding_box\label_me_xml\464.xml"

const dir_path = raw"D:\training_set\gt\bounding_box\label_me_xml"

boxes = Dict()

for xml_file in readdir(dir_path)

    doc = readxml(joinpath(dir_path, xml_file))
    primates = root(doc)

    name = split(xml_file, ".")[1]

    boxes[name] = []

    for poly in findall("//polygon", primates)

        xs = map(findall(".//x", poly)) do x
            parse(Float32,  x.content) |> Int
        end |> x -> (maximum(x), minimum(x))

        ys = map(findall(".//y", poly)) do y
            parse(Float32,  y.content) |> Int
        end |> y -> (maximum(y), minimum(y))

        v, h = ys[2], xs[2]

        hw = xs[1] - xs[2]
        vw = ys[1] - ys[2]

        # NOTE - Resize to 256x256 images
        v, vw = @. (v, vw) / 4000 * 256
        h, hw = @. (h, hw) / 6000 * 256
        
        vw, hw = @. (vw, hw) |> ceil |> Int64
        v, h = @. (v, h) |> round |> Int64

        v, h = check_bounds.([v, h])
        push!(boxes[name], [v, h, vw, hw])
    end

end

# ANCHOR End here, code below not used
# img = raw"D:\training_set\gt\semantic\label_images\464.png"
# img = raw"D:\training_set\images\475.jpg"

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
























