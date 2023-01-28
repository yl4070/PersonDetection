function check_bounds(v)
    if v <= 0
        1
    elseif v >= 256
        256
    else
        v
    end
end

parse_int(x) = parse(Int, x)
Base.round(x::String) = parse(Float64, x) |> round
Base.ceil(x::String) = parse(Float64, x) |> ceil

function Base.round(x::String; digits = 2)
    try
        x = parse(Float32, x)
        round(x; digits = digits)
    catch
        @info "try parsing failed"
    end

    0
end


function get_bboxes(xml_lbl_dir)
    bboxes = Dict()

    # xml_file = readdir(xml_lbl_dir)[1]
    # file = readxml(joinpath(xml_lbl_dir, xml_file))

    for xml_file in readdir(xml_lbl_dir)
        file = readxml(joinpath(xml_lbl_dir, xml_file))
        doc_root = root(file)

        img_name = split(xml_file, ".")[1]
        bboxes[img_name] = []

        img_h = findfirst("//height", doc_root).content |> parse_int
        img_w = findfirst("//width", doc_root).content |> parse_int

        # bbox = findall("//object", doc_root)[1]
        # print(t)

        for bbox in findall("//object", doc_root)
            xmin = findfirst(".//xmin", bbox).content |> round |> Int
            xmax = findfirst(".//xmax", bbox).content |> round |> Int
            ymin = findfirst(".//ymin", bbox).content |> round |> Int
            ymax = findfirst(".//ymax", bbox).content |> round |> Int

            # h = ymax - ymin
            # w = xmax - xmin

            class = findfirst(".//name", bbox).content

            ymin, ymax = @. (ymin, ymax) / img_h * 256
            xmin, xmax = @. (xmin, xmax) / img_w * 256

            xmax, ymax = @. (xmax, ymax) |> ceil |> Int64
            xmin, ymin = @. (xmin, ymin) |> round |> Int64
            xmin, ymin = check_bounds.([xmin, ymin])
            push!(bboxes[img_name], (bbox = [ymin, xmin, ymax, xmax], class = class))
        end
    end
    bboxes
end