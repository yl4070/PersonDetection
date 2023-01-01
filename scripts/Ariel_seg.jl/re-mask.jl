

using Images
using ProgressMeter


const maskdir = raw"D:\training_set\gt\semantic\label_images"
const persondir = raw"D:\training_set\personmasks"

function maskperson(img)

    colorP = RGB((255, 22, 96)./255...)
    color0 = RGB(0,0,0)

    img = imresize(img, (256, 256))

    map(img) do i
        i == colorP ? colorP : color0
    end
end

function generateMask(boxes)
    
    @showprogress for name in keys(boxes)
        img_name = name * ".png"
        img = joinpath(maskdir, img_name) |> Images.load

        save(joinpath(persondir, img_name), maskperson(img))
    end
end

generateMask(boxes)






