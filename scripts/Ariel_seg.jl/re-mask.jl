

using Images
using ProgressMeter


maskdir = raw"D:\training_set\gt\semantic\label_images"

function maskperson(img)

    colorP = RGB((255, 22, 96)./255...)
    color0 = RGB(0,0,0)

    img = imresize(img, (200, 300))

    map(img) do i
        i == colorP ? colorP : color0
    end
end

begin
    persondir = raw"D:\training_set\personmasks"
    @showprogress for name in keys(boxes)
        img_name = name * ".png"
        img = joinpath(maskdir, img_name) |> Images.load

        save(joinpath(persondir, img_name), maskperson(img))
    end
end

