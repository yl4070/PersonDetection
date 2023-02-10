# using Pkg
# Pkg.activate("/home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class")

begin
    using BSON: @save
    using CUDA
    using Dates
    using EzXML
    # using EvidentialFlux
    using FastAI, FastVision
    using FLoops
    using Flux
    using Images
    using JSON3
    using Logging
    using Metalhead
    using MLUtils
    using ProgressMeter
    using Random
    # using SpecialFunctions
    using Statistics
    using TensorBoardLogger
    # using PaddedViews
    using FluxMPI
    import Optimisers
    FluxMPI.Init(verbose=true)
end

include(raw"/home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/getlbl.jl")
include(raw"/home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/get_data.jl")
include(raw"/home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/losses.jl")
include(raw"/home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/model.jl")
include(raw"/home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/training.jl")
include(raw"/home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/read_xml.jl")

const SIZE = 256
const RATIO = 3.5555555555
const V, H = (SIZE, SIZE) .÷ RATIO # REVIEW - check output size later.

CUDA.allowscalar(false)

function main(img_dir, xml_lbl_dir)
    nepoch = 300
    num_filters = 256
    nclass = 20
    img_size = SIZE

    # annotations, imgnames, info_dict = get_info(img_dir, lbl_dir)
    # bboxes_dict = getbbox(annotations, imgnames, info_dict; sz = img_size)
    bboxes_dict = read_xml(xml_lbl_dir, sz = img_size)

    xiter, yiter = get_data(bboxes_dict, img_dir, sz = img_size) 

    # trainset, testset = splitobs((xiter, yiter), at = .9, shuffle = true)

    # @save "testset.bson" testset

    model = centernet(nclass, num_filters, img_size)
    dl = dataloader(xiter, yiter)

    # @info "local rank: $(local_rank())"
    trainevidential(dl, model, nepoch)
end

# xml_lbl_dir = raw"/home/sr8685/ObjectDetection/Datasets/swimcar/Pascal/Train/Annotations"
# xml_img_dir = raw"/home/sr8685/ObjectDetection/Datasets/swimcar/Pascal/Train/JPEGImages"
# main(xml_img_dir, xml_lbl_dir)

# const lbl_dir = raw"D:\Github\PersonDetection\scripts\Keras_CenterNet\Datasets\train.json"
const img_dir = raw"/home/sr8685/PersonDetection/scripts/Datasets/VOCdevkit/VOC2012/JPEGImages"
const xml_lbl_dir = raw"/home/sr8685/PersonDetection/scripts/Datasets/VOCdevkit/VOC2012/Annotations"
main(img_dir, xml_lbl_dir)

# size(y.heatmap)
# size(x) check this
# size(y.heatmap)
