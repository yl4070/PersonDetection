using Flux, FastAI, FastVision
using Metalhead


include("model_blocks.jl") # load unsampling ResBlock  


task = SupervisedTask(
    (FastVision.Image{2}(), Mask{2}(["p","b"])),
    (
        ProjectiveTransforms((64, 64)),
        ImagePreprocessing(),
        OneHot()
    )
)

resbone = Metalhead.ResNet(34).layers[1:end-1]
unet = taskmodel(task, resbone)

down = Chain(make_down(3, 196), make_down(196, 3))
backbone = Chain(down, unet[1][1])

nh = 128
nclass = 80

model = Chain(
    backbone,
    Split(
        Chain(ResBlock(67, nh), convniglayer(nh, 2; ks = 1)), # offset
        Chain(ResBlock(67, nh), convniglayer(nh, 2; ks = 1)), # size
        Chain(ResBlock(67, nh), convxlayer(nh, 1; ks = 1)), # heatmap
        # Chain(ResBlock(67, 3), convxlayer(3, 2; ks = 1)), # segmentation
        Chain(ResBlock(67, nh), convdirlayer(nh, nclass; ks = 1))  # evidential
    )
)
# Total: 274 trainable arrays, 26_993_822 parameters,
# plus 160 non-trainable, 30_368 parameters, summarysize 103.259 MiB.

x = rand(Float32, 256, 256, 3, 10)
yhat = model(x)

yhat[4]

# y = yhat[2]

# hg_mod = Chain(
#     convx(3, 128; stride = 2),
#     ResBlock(128, 96; stride = 2),
#     hourglass(96, 128),
#     hourglass(96, 128), # output channel = 96
#     Split(
#         Chain(convx(96, 64), Conv((1,1), 64 => 1, relu)),
#         Chain(convx(96, 64), Conv((1,1), 64 => 2)),
#         Chain(convx(96, 64), Conv((1,1), 64 => 2))
#     )
# ) |> gpu


