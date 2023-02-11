using Pkg
# Pkg.activate("/home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class")

begin
    import BSON
    using CUDA
    using Dates
    using EzXML
    using FastAI, FastVision
    using FLoops
    using Flux
    using Images
    using JSON3
    using Logging
    using Metalhead
    using MLUtils
    using ProgressMeter
    using JLD2
    using Random
    # using SpecialFunctions
    using Statistics
    using TensorBoardLogger
    # using PaddedViews
    import Optimisers
end

using FluxMPI
FluxMPI.Init(verbose=true)

include(raw"/home/yl4070/Obj_simple/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/get_data.jl")
include(raw"/home/yl4070/Obj_simple/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/losses.jl")
include(raw"/home/yl4070/Obj_simple/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/model.jl")
include(raw"/home/yl4070/Obj_simple/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/training.jl")
include(raw"/home/yl4070/Obj_simple/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/read_xml.jl")

# const SIZE = 256
# const RATIO = 3.5555555555
# const V, H = (SIZE, SIZE) .÷ RATIO # REVIEW - check output size later.

CUDA.allowscalar(false)

function main()

    imsize = 256

    # bboxes_dict = read_xml(xml_lbl_dir, sz = imsize)
    BSON.@load "bbox_dict.bson" bboxes_dict

    # imglist = keys(bboxes_dict) |> collect
    imglist = JLD2.load("imglist.jld2", "imglist")

    ids = ["x$i" for i in 1:17125]

    f = jldopen("xs.jld2", "r")
    xiter = mapobs(ids) do i
        f[i]
    end

    yiter = get_ydata(imglist, bboxes_dict, imsize, 72)

    trd = TrData(xiter, yiter)

    model = centernet(256, imsize)

    dl = dataloader(trd)

    trainevidential(dl, model)
    close(f)
end

# xml_lbl_dir = raw"/home/sr8685/ObjectDetection/Datasets/swimcar/Pascal/Train/Annotations"
# xml_img_dir = raw"/home/sr8685/ObjectDetection/Datasets/swimcar/Pascal/Train/JPEGImages"
# main(xml_img_dir, xml_lbl_dir)

# const lbl_dir = raw"D:\Github\PersonDetection\scripts\Keras_CenterNet\Datasets\train.json"
# const img_dir = "/home/yl4070/PersonDetection/VOCdevkit/VOC2012/JPEGImages"
# const xml_lbl_dir = "/home/yl4070/PersonDetection/VOCdevkit/VOC2012/Annotations"
main()

# jldsave("imglist.jld2"; imglist)

# JLD2.load("imglist.jld2", "imglist")

# @save "imglist.bson" imglist

# # @save "bbox_dict.bson" bboxes_dict

# xdat = get_xdata(bboxes_dict, img_dir; sz = 512)


# xdat = get_xdata(imglist, img_dir; sz = 256)

# using ProgressMeter

# f = jldopen("xs.jld2", "w")
# @showprogress for i in 1:length(xdat)

#     write(f, "x$i", getobs(xdat, i))
# end
# close(f)

# # dat, i_st = iterate(trd)

# xdat = get_xdata(bboxes_dict, img_dir; sz = 512)

