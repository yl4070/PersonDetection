using Flux

# define split util
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x) = map(f -> f(x), m.paths)

function centernet(num_classes, num_filters)

    # num_classes = 20
    input_size = 256
    # freeze_bn = true
    # num_filters = 256

    # output_size = floor(input_size/4)

    resbone = Metalhead.ConMixer(:b4).layers[1:end-1]

    # C5 = resbone.layers[end]
    # x = Chain(C5, Dropout(0.5))
    _, _, last_ch, _ = Flux.outputsize(Chain(resbone, Dropout(0.5)), (input_size,input_size,3,1))
    
    # last_ch = 512
    layers = []
    for i in 0:2
        num_filters = floor(num_filters/(2^i)) |> Int

        conv = Flux.ConvTranspose((4,4), last_ch => num_filters, stride=2, pad=(1,1), bias=false, init=Flux.kaiming_normal)
        bn = BatchNorm(num_filters, Flux.swish)
        push!(layers, conv)
        push!(layers, bn)
        last_ch = num_filters
    end 
    

    backbone = Chain(resbone, Dropout(0.5), layers...)
    
    _, _, nh, _ = Flux.outputsize(backbone, (input_size, input_size, 3, 1))

    hm_head = Chain(Conv((3,3), nh => 64, pad=(1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, Flux.swish),
                    Conv((1, 1), 64 => 1, σ; init = Flux.kaiming_normal))

    wh_head = Chain(Conv((3,3), nh => 64, pad=(1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, Flux.swish),
                    Conv((1, 1), 64 => 2, init = Flux.kaiming_normal))

    off_head = Chain(Conv((3,3), nh => 64, pad= (1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, Flux.swish),
                    Conv((1, 1), 64 => 2, init = Flux.kaiming_normal))

    class_head = Chain(Conv((3,3), nh => 64, pad= (1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, Flux.swish),
                    Conv((1, 1), 64 => 20, init = Flux.kaiming_normal))

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


