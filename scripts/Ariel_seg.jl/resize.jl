


const img_dir = raw"D:\training_set\images"
const small_dir = raw"D:\training_set\small"


let 
    @showprogress for img in readdir(img_dir)
        
        im = joinpath(img_dir, img) |> load

        im = imresize(im, (256, 256))

        save(joinpath(small_dir, img), im)
    end
end





