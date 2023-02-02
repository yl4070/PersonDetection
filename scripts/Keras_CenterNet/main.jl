using Pkg
Pkg.activate("D:/Github/PersonDetection")

using BSON: @save
using CUDA
using Dates
using EzXML
using EvidentialFlux
using FastAI, FastVision
using Flux
using Images
using Logging
using Metalhead
using MLUtils
using ProgressMeter
using Random
using SpecialFunctions
using Statistics
using TensorBoardLogger
using PaddedViews

include("getlbl.jl") # get bounding_box info from XML files
include("get_data.jl") # load and encoding images and lables, return xiter, and yiter
include("losses.jl")
include("model.jl")
include("training.jl")

const SIZE = 512
const RATIO = 3.5 
const V, H = (SIZE, SIZE) .÷ RATIO # REVIEW - check output size later.

CUDA.allowscalar(false)

function main(img_dir, lbl_dir)
    nepoch = 80
    num_filters = 256
    nclass = 20
    img_size = SIZE

    annotations, imgnames, info_dict = get_info(img_dir, lbl_dir)
    bboxes_dict = getbbox(annotations, imgnames, info_dict; sz = img_size)

    xiter, yiter = get_data(bboxes_dict, img_dir, sz = img_size) 

    model = centernet(nclass, img_size)
    dl = dataloader(xiter, yiter)

    trainevidential(dl, model, nepoch)
end

# xml_lbl_dir = raw"/home/sr8685/ObjectDetection/Datasets/swimcar/Pascal/Train/Annotations"
# xml_img_dir = raw"/home/sr8685/ObjectDetection/Datasets/swimcar/Pascal/Train/JPEGImages"
# main(xml_img_dir, xml_lbl_dir)

const lbl_dir = raw"D:\Github\PersonDetection\scripts\Keras_CenterNet\Datasets\train.json"
const img_dir = raw"D:\Github\PersonDetection\scripts\Keras_CenterNet\Datasets\JPEGImages"
main(img_dir, lbl_dir)

# size(y.heatmap)
# size(x) check this
# size(y.heatmap)
