


const img_dir = raw"D:\training_set\images"
const small_dir = raw"D:\training_set\small"


let 
    @showprogress for img in readdir(img_dir)
        
        im = joinpath(img_dir, img) |> load

        im = imresize(im, (256, 256))

        save(joinpath(small_dir, img), im)
    end
end




# ANCHOR Resize COCO dataset

let coco_dir = raw"D:\train2017\train2017", coco_small_dir = raw"D:\COCO\small"

    @showprogress for img in readdir(coco_dir)
        
        im = joinpath(coco_dir, img) |> load

        im = imresize(im, (256, 256))

        save(joinpath(coco_small_dir, img), im)
    end
end




