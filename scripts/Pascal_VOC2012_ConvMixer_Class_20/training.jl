function dataloader(xiter, yiter, batchsize = 8*total_workers(), collate = true, shuffle = true, partial = false, parallel = true)

    # clean_println("len of xiter: $(length(xiter))")

    trainx = DistributedDataContainer(xiter)
    trainy = DistributedDataContainer(yiter)

    # clean_println("len of trainx: $(length(trainx))")
    # clean_println("ids of $(local_rank()) are $(trainx.idxs)")

    # clean_println("threads = $(ThreadedEx())")
    x_data = BatchView(shuffleobs(trainx); batchsize=batchsize ÷ total_workers(),
                            partial=false, collate=true)
    y_data = BatchView(shuffleobs(trainy); batchsize=batchsize ÷ total_workers(),
                            partial=false, collate=true)

    xdistiter = Iterators.cycle(MLUtils.eachobsparallel(x_data; executor=ThreadedEx(), buffer=true))
    ydistiter = Iterators.cycle(MLUtils.eachobsparallel(y_data; executor=ThreadedEx(), buffer=true))

    # DataLoader((trainx, trainy), batchsize = batchsize ÷ total_workers() , collate = collate, shuffle = shuffle, partial = partial, parallel = parallel)
    # DataLoader((xiter, yiter), batchsize = 2, collate = collate, shuffle = shuffle, partial = partial, parallel = parallel)
    xdistiter,  ydistiter
end

# ANCHOR evidential training
function trainevidential(dl, model, nepoch = 20)

    xdist, ydist = dl

    _, x_st = iterate(xdist)
    _, y_st = iterate(ydist)



    dt = replace(string(now()),r":|\."=>"-")
    data = "VOC2012"
    model_nm = "5NodeBatch_MixCls"
    folder = "Pascal_VOC2012_ConvMixer_Class_20"
    mod_dt = data*"_"*model_nm*"_"*dt

    logger = TBLogger("/home/sr8685/PersonDetection/scripts/$folder/TensorBoardLogs/$mod_dt", min_level=Logging.Info)

    model = FluxMPI.synchronize!(model |> gpu)
    # ps = Flux.params(model)

    # opt = Flux.Optimise.Optimiser(ClipNorm(1.0), Adam())
    opt_st = Optimisers.setup(Optimisers.Adam(), model)
    opt_st = FluxMPI.synchronize!(opt_st)

    for s in 1:80_000

        x, x_st = iterate(xdist, x_st)
        y, y_st = iterate(ydist, y_st)
 
        @show "Step $s"
        t_start = time()

        local loss, Lk, Loff, Lsz, Lclass, Lclsheat, ŷ

        x, y = x |> gpu, y |> gpu
        pos_idx = f32(y.heatmap .== 1)
        
        loss, gs = Flux.withgradient(model) do m
            
            ŷ = m(x)

            Lk = loss_k(ŷ[1], y.heatmap, pos_idx)
            Lsz = pointsloss(ŷ[2], y.size, pos_idx)
            Loff = pointsloss(ŷ[3], y.off, pos_idx)
            Lclsheat = loss_k(sum(ŷ[4]; dims = 3), y.heatmap, pos_idx)
            Lclass = pointsentropy(ŷ[4], y.class, pos_idx)
            
            loss = Lk + Loff + .1Lsz + Lclass + Lclsheat
            loss / total_workers()
        end
        
        # @show loss, Lk, Loff, Lsz, Lclass, Lclsheat
        clean_println("epoch $s: loss = $loss, lk = $Lk, lsz = $Lsz, lcls = $Lclass")

        gs1 = FluxMPI.allreduce_gradients(gs[1])

        Optimisers.update!(opt_st, model, gs1)
    
        with_logger(logger) do 
            @info "train" loss=loss Lk=Lk Loff=Loff Lsz=Lsz Lclass=Lclass Lclsheat=Lclsheat
            @info "train_progress" epoch = s
        end

        if s % 5000 == 0
            let model = cpu(model)
                folder_path = "/home/sr8685/PersonDetection/scripts/$folder/models/modellog_$mod_dt"
                if !isdir(folder_path)
                    mkdir(folder_path)
                end
                # @save "modellog/modelswim_b1-$(now())-ep$e.bson" model
                d = replace(string(now()),r":|\."=>"-")
                @save "/home/sr8685/PersonDetection/scripts/$folder/models/modellog_$mod_dt/model-$model_nm-$(d)_step$s.bson" model
                # @save "/home/sr8685/ObjectDetection/PersonDetection/scripts/pascal/modellog_nig/modelpascal_b1_nig-$(now())-ep$e.bson" model
            end
        end
        t_end = time()
        time_taken = t_end - t_start
        clean_println("Time taken: $time_taken")
    end

    model = model |> cpu
    d = replace(string(now()),r":|\."=>"-")
    @save "/home/sr8685/PersonDetection/scripts/$folder/models/modellog_$mod_dt/model-$model_nm-$(d)_full.bson" model
    # @save "modelpascal_all_nig_$mod_dt.bson" model
end