
# ANCHOR building blocks
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

struct ResBlock
    convs::Any
    idconv::Any
    pool::Any
end

function ResBlock(ni::Int, nf::Int; stride::Int = 1, ndim = 2)

    layers = [
        convxlayer(ni, nf; stride = stride, ndim = ndim),
        convxlayer(nf, nf; zero_bn = true, act = false, ndim = ndim)
    ]

    connect = if stride > 1 || nf != ni
        convxlayer(ni, nf; stride = stride)
    else
        identity
    end

    ResBlock(Chain(layers...), connect, identity)
                    # stride == 1 ? identity : MeanPool(ntuple(_ -> 2, ndim))) # REVIEW - do we pool?
end
Flux.@functor ResBlock
(r::ResBlock)(x) = act_fn.(r.convs(x) .+ r.idconv(r.pool(x)))

# down sampling layer
make_down(ni, nf) = Chain(ResBlock(ni, nf; stride = 2), Dropout(.1), ResBlock(nf, nf))


# ANCHOR DIR layer
diract(x) = Flux.softplus(x) + 1

function convdirlayer(ni, nf; ks = 3, stride = 1, ndim = 2)
    # bn = BatchNorm(nf, diract) # REVIEW No batchnorm for dir loss

    convx(ni, nf, diract; ks = ks, stride = stride, ndim = ndim)
end


# ANCHOR NIG layer
function convniglayer(ni, nf; ks = 3, stride = 1, ndim = 2)
    # bn = BatchNorm(nf, diract) # REVIEW No batchnorm for dir loss

    convx(ni, 4nf, identity; ks = ks, stride = stride, ndim = ndim)
end










# define split util
struct Split{T}
    paths::T
end
Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x) = map(f -> f(x), m.paths)

