function dataloader(dataiter, batchsize = 8*total_workers(), collate = true, shuffle = true, partial = false, parallel = true)

    # clean_println("len of xiter: $(length(xiter))")


    traindata = DistributedDataContainer(dataiter)

    # clean_println("len of trainx: $(length(trainx))")
    # clean_println("ids of $(local_rank()) are $(trainx.idxs)")

    # clean_println("threads = $(ThreadedEx())")
    dataview = BatchView(shuffleobs(traindata); batchsize=batchsize ÷ total_workers(), partial=false, collate=true)


    distiter = Iterators.cycle(MLUtils.eachobs(dataview))

    # DataLoader((trainx, trainy), batchsize = batchsize ÷ total_workers() , collate = collate, shuffle = shuffle, partial = partial, parallel = parallel)
    # DataLoader((xiter, yiter), batchsize = 2, collate = collate, shuffle = shuffle, partial = partial, parallel = parallel)
    distiter
end

# ANCHOR evidential training
function trainevidential(dl, model)

    trdist = dl

    _, iter_st = iterate(trdist)

    dt = replace(string(now()),r":|\."=>"-")
    data = "VOC2012"
    model_nm = "HConvMix"
    # folder = "Pascal_VOC2012_ConvMixer_Class_20"
    mod_dt = data*"_"*model_nm*"_"*dt

    logger = TBLogger("/home/yl4070/PersonDetection/scripts/TensorBoardLogs/$mod_dt", min_level=Logging.Info)

    model = FluxMPI.synchronize!(model |> gpu)

    opt_st = Optimisers.setup(Optimisers.Adam(), model)
    opt_st = FluxMPI.synchronize!(opt_st)

    for s in 1:80_000

        (x, y), iter_st = iterate(trdist, iter_st)
 
        @show "Step $s"
        t_start = time()

        x, y = x |> gpu, y |> gpu

        local Lk, Lsz, Loff

        pos_idx = f32(y.heatmap .== 1)
        flt_idx = sum(pos_idx; dims = 3)
            
        loss, gs = Flux.withgradient(model) do m
            
            ŷ = m(x)
            # loss = part_loss(ŷ, y)

            Lk = loss_k(ŷ[1], y.heatmap, pos_idx)
            Lsz = pointsloss(ŷ[2], y.size, flt_idx)
            Loff = pointsloss(ŷ[3], y.off, flt_idx)

            loss = Lk + Lsz + Loff
            loss / total_workers()
        end
        
        # @show loss, Lk, Loff, Lsz, Lclass, Lclsheat
        clean_println("epoch $s: loss = $loss, lk = $Lk")

        gs1 = FluxMPI.allreduce_gradients(gs[1])

        Optimisers.update!(opt_st, model, gs1)
    
        if local_rank() == 1
            with_logger(logger) do 
                @info "train" loss=loss Lk = Lk Lsz = Lsz Loff = Loff
                @info "train_progress" step = s
            end
        end

        if s % 5000 == 0 && local_rank() == 1
            let model = cpu(model)
                folder_path = "/home/yl4070/PersonDetection/scripts/modellog_$mod_dt"
                if !isdir(folder_path)
                    mkdir(folder_path)
                end

                d = replace(string(now()),r":|\."=>"-")

                BSON.@save joinpath(folder_path, "model-$model_nm-$(d)_step$s.bson") model
            end
        end

        t_end = time()

        time_taken = t_end - t_start
        clean_println("Time taken: $time_taken")
    end

    if local_rank() == 1
        model = model |> cpu
        d = replace(string(now()),r":|\."=>"-")
        BSON.@save "/home/yl4070/PersonDetection/scripts/modellog_$mod_dt/model-$model_nm-$(d)_full.bson" model
    end
end








