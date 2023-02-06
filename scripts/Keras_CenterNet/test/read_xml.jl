function check_bound(v; sz = 256)
    if v <= 0
        1
    elseif v >= sz
        sz
    else
        v
    end
end

function intify(x)
    round(x) |> Int
end

parseInt(x) = parse(Int, x)

Base.round(x::String) = parse(Float64, x) |> round

Base.ceil(x::String) = parse(Float64, x) |> ceil

function read_xml(xml_lbl_dir; sz=256)
    bboxes = Dict()

    for xml_file in readdir(xml_lbl_dir)
        doc = readxml(joinpath(xml_lbl_dir, xml_file))
        file = root(doc)

        imgname = findfirst("//filename", file).content
        bboxes[imgname] = [] 

        ih = findfirst("//size/width", file).content |> parseInt
        iv = findfirst("//size/height", file).content |> parseInt

        objects = findall("//object", file)
        for obj in objects
            class = findfirst(".//name", obj).content
            difficulty = findfirst(".//difficult", obj).content |> parseInt
            xmin = findfirst(".//bndbox/xmin", obj).content |> parseInt
            ymin = findfirst(".//bndbox/ymin", obj).content |> parseInt
            xmax = findfirst(".//bndbox/xmax", obj).content |> parseInt
            ymax = findfirst(".//bndbox/ymax", obj).content |> parseInt
            hw = xmax - xmin # horizontal width
            vw = ymax - ymin # vertical width (height)

            mx =  iv > ih ? sz / iv : sz/ih # mx => multiplier
            v = intify(ymin * mx)
            h = intify(xmin * mx)
            vw, hw = intify.((vw * mx, hw * mx))
            v, h = check_bound.([v, h]; sz = sz)   

            push!(bboxes[imgname], (bbox = [v, h, vw, hw], class = class, difficulty = difficulty))
        end
    end

    bboxes
end