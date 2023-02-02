function dataloader(xiter, yiter, batchsize = 6, collate = true, shuffle = true, partial = false, parallel = false)
    DataLoader((xiter, yiter), batchsize = batchsize, collate = collate, shuffle = shuffle, partial = partial, parallel = parallel, rng=MersenneTwister(2))
end

# ANCHOR evidential training
function trainevidential(dl, model, nepoch = 20)
    dt = replace(string(now()),r":|\."=>"-")
    data = "pascal"
    model_nm = "ConvMixer"
    mod_dt = model_nm*"_"*dt

    # logger = TBLogger("TensorBoardLogs/pascal_nig_20230119_2230", min_level=Logging.Info)
    logger = TBLogger("D:/Github/PersonDetection/scripts/Keras_CenterNet/TensorBoardLogs/$data-$mod_dt", min_level=Logging.Info)

    model = model |> gpu
    # ps = Flux.params(model)
    ps = Flux.params(model)

    # gs[ps[2]]
    # opt = Flux.Optimise.Optimiser(ClipNorm(1.0), Adam())
    opt = Adam()
    
    for e in 1:nepoch

        @show "Epoch $e"

        for (x, y) in dl
            
            local loss, Lk, Loff, Lsz, Ldir, ŷ

            x, y = x |> gpu, y |> gpu
            
            pos_idx = f32(y.heatmap .== 1)
            clf_idx = sum(pos_idx, dims = 3)
            
            gs = gradient(ps) do 
                
                ŷ = model(x)

                # yhat_pos = f32(ŷ[3] .> .3)

                Lk = loss_k(ŷ[1], y.heatmap, pos_idx)
                Loff = pointsloss(ŷ[3], y.off, clf_idx)
                Lsz = pointsloss(ŷ[2], y.size, clf_idx)

                loss = Lk + Loff + .1Lsz 
                loss 
            end
            
            @show loss, Lk, Loff, Lsz#, Ldir

            Flux.update!(opt, ps, gs)
        
            with_logger(logger) do 
                @info "train" loss=loss Lk=Lk Loff=Loff Lsz=Lsz #Ldir=Ldir
                @info "train_progress" epoch = e
            end
        end

        if e % 20 == 0
            let model = cpu(model)
                folder_path = "D:/Github/PersonDetection/scripts/Keras_CenterNet/modellog_$mod_dt"
                if !isdir(folder_path)
                    mkdir(folder_path)
                end
                # @save "modellog/modelswim_b1-$(now())-ep$e.bson" model
                d = replace(string(now()),r":|\."=>"-")
                @save "D:/Github/PersonDetection/scripts/Keras_CenterNet/modellog_$mod_dt/model$data-$model_nm-$(d)_ep$e.bson" model
                # @save "/home/sr8685/ObjectDetection/PersonDetection/scripts/pascal/modellog_nig/modelpascal_b1_nig-$(now())-ep$e.bson" model
            end
        end
    end

    model = model |> cpu
    @save "D:/Github/PersonDetection/scripts/Keras_CenterNet/modellog_$mod_dt/model$data-full_$mod_dt.bson" model
    # @save "modelpascal_all_nig_$mod_dt.bson" model
end