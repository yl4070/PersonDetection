using CairoMakie
using BSON: @save
using ProgressMeter

include("loading_lbl.jl") # get bounding_box info from XML files
include("get_data.jl") # load and encoding images and lables, return xiter, and yiter

include("model.jl")
include("losses.jl")

seg_lossfn = tasklossfn(task)
dl = DataLoader((xiter, yiter), batchsize = 10, collate = true, shuffle = true, partial = true)

function train(dl, model, nepoch = 20)
    
    model = model |> gpu
    ps = Flux.params(model)

    opt = Adam()

    for e in 1:nepoch

        @info "Epoch $e"

        i = 0
        @showprogress for (x, y) in dl

            local loss

            x, y = x |> gpu, y |> gpu

            pos_idx = f32(y.heatmap .== 1)

            gs = gradient(ps) do 
                
                ŷ = model(x)

                Lk = loss_k(ŷ[3], y.heatmap, pos_idx)
                Lsz = pointsloss(ŷ[2], y.size, pos_idx)
                Loff = pointsloss(ŷ[1], y.off, pos_idx)
                Lseg = seg_lossfn(ŷ[4], y.mask)

                loss = Lk + Lseg + Loff + .1Lsz
                loss
            end

            Flux.update!(opt, ps, gs)
        
            if i % 10 == 0 
                @show loss
            end

            i+=1
        end

    end

    model = model |> cpu
    @save "model2.bson" model
end

train(dl, model, 500)

# ANCHOR evidential training
function trainevidential(dl, model, nepoch = 20)
    
    model = model |> gpu
    ps = Flux.params(model)

    opt = Adam()

    for e in 1:nepoch

        @info "Epoch $e"

        i = 0
        @showprogress for (x, y) in dl

            local loss

            x, y = x |> gpu, y |> gpu

            pos_idx = f32(y.heatmap .== 1)

            gs = gradient(ps) do 
                
                ŷ = model(x)
                Lk = loss_k(ŷ[3], y.heatmap, pos_idx)

                # Loff = pointsloss(ŷ[1], y.off, pos_idx)
                Lsz = pointnig(nigloss2(y[1], y.off), pos_idx)
                # Lsz = pointsloss(ŷ[2], y.size, pos_idx)
                Lsz = pointnig(nigloss2(ŷ[2], y.size), pos_idx)

                # Lseg = seg_lossfn(ŷ[4], y.mask)
                Ldir = convdirloss(ŷ[4], y.class, e)

                loss = Lk + Loff + .1Lsz + Ldir
                loss
            end

            Flux.update!(opt, ps, gs)
        
            if i % 10 == 0 
                @show loss
            end

            i+=1
        end

    end

    model = model |> cpu
    @save "model128ev.bson" model
end

trainevidential(dl, model, 1_500)


# ANCHOR training code end here 

using BSON: @load

using Random
@load "model128nbn.bson" model

xtest = getobs(xiter, 40)

CairoMakie.heatmap(xtest[:,:,3])

yy = getobs(yiter, 40)

CairoMakie.heatmap(yy.mask[:,:,2])

xtest = Flux.unsqueeze(xtest, 4)

yhat = model(xtest)


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








































