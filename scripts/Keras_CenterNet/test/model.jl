using Flux

# define split util
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x) = map(f -> f(x), m.paths)

function centernet(num_classes, num_filters, input_size)

    # num_classes = 20
    # input_size = 128
    # freeze_bn = true
    # num_filters = 256
    # output_size = floor(input_size/4)

    resbone = Metalhead.ConvMixer(:base).layers[1:end-1]

    _, _, last_ch, _ = Flux.outputsize(Chain(resbone, Dropout(0.5)), (input_size,input_size,3,1))
    
    # nh = last_ch
    # last_ch = 512

    uplayer = Chain(
        conv = Flux.ConvTranspose((4,4), last_ch => num_filters, stride=2, pad=(1,1), bias=false, init=Flux.kaiming_normal),
        bn = BatchNorm(num_filters, Flux.swish)
    )
    

    backbone = Chain(resbone, Dropout(0.5), uplayer)
    
    _, _, nh, _ = Flux.outputsize(backbone, (input_size, input_size, 3, 1))

     
    # backbone = Chain(resbone, Dropout(0.5), layers...)
   
    hm_head = Chain(Conv((3,3), nh => 64, pad=(1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, relu),
                    Conv((1, 1), 64 => 1, σ; init = Flux.kaiming_normal))

    wh_head = Chain(Conv((3,3), nh => 64, pad=(1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, relu),
                    Conv((1, 1), 64 => 2, init = Flux.kaiming_normal))

    off_head = Chain(Conv((3,3), nh => 64, pad= (1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, relu),
                    Conv((1, 1), 64 => 2, init = Flux.kaiming_normal))

    class_head = Chain(Conv((3,3), nh => 64, pad= (1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, Flux.swish),
                    Conv((1, 1), 64 => num_classes, init = Flux.kaiming_normal))

    Chain(
        backbone,
        Split(
            hm_head, 
            wh_head,
            off_head,
            class_head
        )
    )
end

# centernet(num_classes, input_size)


