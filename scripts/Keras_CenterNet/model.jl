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
    # input_size = 256
    # freeze_bn = true
    # num_filters = 256
    # output_size = floor(input_size/4)

    resbone = Metalhead.ConvMixer(:base).layers[1:end-1]

    _, _, last_ch, _ = Flux.outputsize(Chain(resbone, Dropout(0.5)), (input_size,input_size,3,1))
    
    nh = last_ch

    hm_head = Chain(Conv((3,3), nh => 64, pad=(1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, relu),
                    Conv((1, 1), 64 => num_classes, σ; init = Flux.kaiming_normal))

    wh_head = Chain(Conv((3,3), nh => 64, pad=(1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, relu),
                    Conv((1, 1), 64 => 2, init = Flux.kaiming_normal))

    off_head = Chain(Conv((3,3), nh => 64, pad= (1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, relu),
                    Conv((1, 1), 64 => 2, init = Flux.kaiming_normal))

    Chain(
        backbone,
        Split(
            hm_head, 
            wh_head,
            off_head
        )
    )
end

# centernet(num_classes, input_size)


