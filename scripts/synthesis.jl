using Images
using ProgressMeter
using BSON: @save

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
   sz = (97, 200)

   person = imresize(person, )
   x, y = rand(200:4000, 1)[1], rand(200:6000, 1)[1]
   w, h = size(person) .÷ 2

   for i in x-w:x+w-1, j in y-h:y+h-1
      
      a, b = i-x+w+1, j-y+h+1
      img[i, j] = person[a, b] == color0 ? img[i,j] : person[a,b]
   end
end


# ANCHOR add person
function AttachPerson!(img::AbstractArray, coords::Tuple{Vararg{T} where T <: Integer}, person::AbstractArray, mask::AbstractArray) 

   x, y = coords
   w, h = size(person) .÷ 2

   for i in x-w:x+w-1, j in y-h:y+h-1
      
      a, b = i-x+w+1, j-y+h+1
      img[i, j] = person[a, b] == color0 ? img[i,j] : person[a,b]
      mask[i,j] = person[a, b] == color0 ? mask[i,j] : personC
   end  
   for i in eachindex(mask)
      mask[i] = mask[i] == personC ? personC : color0
   end
   img
end

function AttachPerson!(x...)
   @error "Wrong inputs"
end

#ANCHOR - 
const sz = (400, 600)
shrink(p) = imresize(p, sz)
tinify(p) = imresize(p, sz .÷ 20)
tinify(p) = imresize(p, sz .÷ 20)
microfy(p) = imresize(p, sz .÷ 30)


pic = rand(pics, 1)[1]
maskp = replace(pic, "jpg" => "png")
img = joinpath(orignal, pic)
mask = joinpath(maskdir, maskp)

mask = load(mask) |> shrink
img = load(img) |> shrink


persons = joinpath.("imgs", rand(peoples, 10)) .|> load .|> tinify
person = person |> tinify
person = person |> microfy


traindir = joinpath(projectdir(), "training", "data")
labeldir = joinpath(projectdir(), "training", "mask")


@showprogress for pic in pics[1:100]

   maskp = replace(pic, "jpg" => "png")
   img = joinpath(orignal, pic)
   mask = joinpath(maskdir, maskp)

   mask0 = load(mask) |> shrink
   img0 = load(img) |> shrink

   for i in 1:10
      img = copy(img0)
      mask = copy(mask0)

      persons = joinpath.("imgs", rand(peoples, 10)) .|> load .|> tinify
      more_persons = joinpath.("imgs", rand(peoples, 10)) .|> load .|> microfy

      persons = vcat(persons, more_persons)

      for person in persons
         pos = (rand(11:389, 1), rand(16:584, 1)) .|> first
         AttachPerson!(img, pos, person, mask)
      end

      name = "$(traindir)/$(pic[1:end-4])_$i.png"
      save(name, img)
      name = "$(labeldir)/$(pic[1:end-4])_$i.png"
      save(name, mask)
   end
end






# ANCHOR training model
using FastAI, FastVision, Flux, Metalhead
using Images

classes = ["bg", "person"]

function load_mask(file)

   mask = loadfile(file)
   imask = map(mask) do x
      x == color0 ? 1 : 2   
   end

   Images.IndirectArray(imask, ["bg", "person"])
end

masks = FastAI.Datasets.loadfolderdata(
    labeldir,
    filterfn=FastVision.isimagefile,
loadfn= f -> load_mask(f))

images = FastAI.Datasets.loadfolderdata(
   traindir, 
   filterfn=FastVision.isimagefile,
loadfn=loadfile)

data = (images, masks)

image, mask = sample = getobs(data, 1)


task = SupervisedTask(
    (FastVision.Image{2}(), Mask{2}(classes)),
    (
      ProjectiveTransforms((256, 256)),
      ImagePreprocessing(),
      OneHot()
    )
)

checkblock(task.blocks.sample, sample)

xs, ys = FastAI.makebatch(task, data, 1:3)

backbone = Metalhead.ResNet(18).layers[1:end-1]
model = taskmodel(task, backbone)
lossfn = tasklossfn(task)

traindl, validdl = taskdataloaders(data, task, 8)

opt = Adam()

learner = Learner(model, lossfn; callbacks=[ToGPU()], data = (traindl, validdl), optimizer = opt)

fitonecycle!(learner, 1, 0.033)


enc = ImagePreprocessing()

encx = encode(enc, Training(), FastVision.Image{2}(), im)

yhat = m(Flux.unsqueeze(encx, 4) |> gpu)





















