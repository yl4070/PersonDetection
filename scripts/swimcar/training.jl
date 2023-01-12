using CairoMakie
using BSON: @save
using ProgressMeter
using CUDA

include("loading_lbl.jl") # get bounding_box info from XML files
include("get_data.jl") # load and encoding images and lables, return xiter, and yiter

include("model.jl")
include("losses.jl")

dl = DataLoader((xiter, yiter), batchsize = 10, collate = true, shuffle = true, partial = true, parallel = true)


# ANCHOR evidential training
function trainevidential(dl, model, nepoch = 20)
    
    model = model |> gpu
    ps = Flux.params(model)

    opt = Adam()

    for e in 1:nepoch

        @info "Epoch $e"

        i = 0
        @showprogress for (x, y) in dl

            local loss, Lk, Loff, Lsz, Ldir

            x, y = x |> gpu, y |> gpu

            pos_idx = f32(y.heatmap .== 1)

            gs = gradient(ps) do 
                
                ŷ = model(x)
                Lk = loss_k(ŷ[3], y.heatmap, pos_idx)

                Loff = pointsloss(ŷ[1], y.off, pos_idx)
                # Loff = pointnig(nigloss2(ŷ[1], y.off), pos_idx)
                Lsz = pointsloss(ŷ[2], y.size, pos_idx)
                # Lsz = pointnig(nigloss2(ŷ[2], y.size), pos_idx)

                # Lseg = seg_lossfn(ŷ[4], y.mask)
                Ldir = pointsdir(ŷ[4], y.class, e, pos_idx)
                # Ldir = convdirloss(ŷ[4], y.class, e)

                loss = Lk + Loff + .1Lsz + Ldir
                loss
            end

            Flux.update!(opt, ps, gs)
        
            if i % 10 == 0 
                @show loss
                @show Lk, Loff, Lsz, Ldir
            end

            i+=1
        end

        if e % 10 == 0
            model = model |> cpu
            @save "modelswim.bson" model
            GC.gc()
            CUDA.reclaim()
            model = model |> gpu
            @info "model saved"
        end
    end

    model = model |> cpu
    @save "modelswim.bson" model
end

trainevidential(dl, model, 100)


# ANCHOR training code end here 

using BSON: @load

using Random
@load "modelpascal.bson" model

xtest = getobs(xiter, 34)

CairoMakie.heatmap(xtest[:,:,3])

yy = getobs(yiter, 34)

CairoMakie.heatmap(yy.heatmap[:,:,1])

xtest = Flux.unsqueeze(xtest, 4)

yhat = model(xtest)


loss_k(yhat[3], yy.heatmap, yy.heatmap .== 1)


CairoMakie.heatmap(yhat[4][:,:,1, 1])
CairoMakie.heatmap(yhat[3][:,:,1, 1])

ol = 1 .- channelview(Gray.(m)) |> Array 

m = joinpath(persondir, "001.png") |> load


hp = yhat[3][:,:,1, 1]

idx = findall(el -> el > .45, hp)


ev = yhat[4][:,:,:, 1]


using EvidentialFlux

o = rand(Float32, 64, 64, 2, 1)
y = rand(Float32, 64, 64, 2, 1)








































