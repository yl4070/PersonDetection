using Images
using ProgressMeter


# ANCHOR extract people
orignal = raw"F:\SemArial\dataset\semantic_drone_dataset\original_images"
maskdir = raw"F:\SemArial\RGB_color_image_masks\RGB_color_image_masks"

personC = RGB{N0f8}(((255, 22, 96)./255)...)

pics = readdir(orignal)
color0 = RGB{N0f8}(0)


@showprogress for pic in pics

   maskp = replace(pic, "jpg" => "png")
   img = joinpath(orignal, pic)
   mask = joinpath(maskdir, maskp)

   mask = load(mask)
   img = load(img)

   img0 = similar(img)
   for i in eachindex(mask)
      img[i] = mask[i] == personC ? img[i] : color0
      img0[i] = mask[i] == personC ? personC : color0
   end


   labels = label_components(img0, trues(3,3))
   bboxes = component_boxes(labels)[3:end]

   imgs = []

   for (p1, p2) in bboxes

      if abs(p1[1]-p2[1]) > 10
         push!(imgs, img[p1[1]:p2[1], p1[2]:p2[2]])   
      end
   end

   for (i,img) in enumerate(imgs)

      name = "$(projectdir())/imgs/$(pic[1:end-4])_$i.png"
      save(name, img)
   end

end




# ANCHOR Generate image
peoples = readdir("imgs")

pic = rand(pics, 1) |> first
img = load(joinpath(orignal, pic))


for _ in 1:rand(10:20, 1)[1]

   person = load(joinpath("imgs", rand(peoples, 1)[1]))
   x, y = rand(200:4000, 1)[1], rand(200:6000, 1)[1]
   w, h = size(person) .÷ 2

   for i in x-w:x+w-1, j in y-h:y+h-1
      
      a, b = i-x+w+1, j-y+h+1
      img[i, j] = person[a, b] == color0 ? img[i,j] : person[a,b]
   end
end

img

















