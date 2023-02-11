
const act_fn = relu

function convx(ni, nf, act = identity; ks = 3, stride = 1, ndim = 2)

    Conv(ntuple(_ -> ks, ndim),
         ni => nf, act,
         stride = stride,
         pad = ks ÷ 2,
         init = Flux.kaiming_normal)
end

function convxlayer(ni, nf; ks = 3, stride = 1, zero_bn = false, act = true, ndim = 2)
    bn = BatchNorm(nf, act ? act_fn : identity)  #REVIEW - do we batchnorm?
    fill!(bn.γ, zero_bn ? 0 : 1)

    Chain(convx(ni, nf; ks = ks, stride = stride, ndim = ndim), bn)
end

struct Split{T}
    paths::T
end
Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x) = map(f -> f(x), m.paths)

function createhead(nh; scale = "up", num_classes = 20)

    if scale == "up"
        conn = PixelShuffle(2)
    elseif scale == "down"
        conn = convxlayer(nh*4, nh; stride = 2)
    else
        conn = identity
        nh = nh * 4
    end

    hm_head = Chain(Conv((3,3), nh => 64, pad=(1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, relu),
                    Conv((1, 1), 64 => num_classes, σ; init = Flux.kaiming_normal))

    wh_head = Chain(Conv((3,3), nh => 64, pad=(1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, relu),
                    Conv((1, 1), 64 => 2, init = Flux.kaiming_normal))

    off_head = Chain(Conv((3,3), nh => 64, pad= (1, 1), bias=false, init=Flux.kaiming_normal),
                    BatchNorm(64, relu),
                    Conv((1, 1), 64 => 2, init = Flux.kaiming_normal))

    head = Split(
            hm_head,
            wh_head, 
            off_head
        )

    Chain(
        conn,
        head
    )
end


function get_hcenter(nh = 256)
    
    backbone = ConvMixer(:large).layers[1:end-1][1]

    # backback = backbone[1]
    # num_classes = 20

    m = Chain(
        backbone[1:15],
        Split(
            createhead(nh; scale = "down"), # y[1] - 37*37
            Chain(backbone[16:19],
                Split(
                    createhead(nh; scale = ""), # y[2][1] 73*73
                    Chain(backbone[20:22], createhead(nh; scale = "up")) # y[2][2] 146*146
                )
            )
        )
    )
    # Total: 209 trainable arrays, 24_239_816 parameters,
    # plus 100 non-trainable, 85_120 parameters, summarysize 92.840 MiB.

    m
end

# x = rand(Float32, 256, 256, 3, 1)

# # x = Flux.unsqueeze(x, dims = 4)

# y = model(x)


# y[1][1]
# y[2][1][1]
# y[2][2][1]



# Flux.outputsize(backbone[1:15], (512, 512, 3, 1))

