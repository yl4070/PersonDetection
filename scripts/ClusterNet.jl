using Flux

x = rand(Float32, 1344, 756, 3, 10) |> gpu

m = Chain(Conv((3, 3), 3 => 3, x -> leakyrelu(x, 0.01); stride = 2),
          BatchNorm(3), MaxPool((2,2)),
          Conv((3, 3), 3 => 3, x -> leakyrelu(x, 0.01); stride = 2),
          BatchNorm(3), MaxPool((2,2)),  
          Conv((3, 3), 3 => 3, x -> leakyrelu(x, 0.01)),
          BatchNorm(3), 
          Conv((1, 1), 3 => 3, x -> leakyrelu(x, 0.01)),
          BatchNorm(3), 
          Conv((3, 3), 3 => 3, x -> leakyrelu(x, 0.01)),
          BatchNorm(3), MaxPool((2,2)),  
          Conv((3, 3), 3 => 3, x -> leakyrelu(x, 0.01)),
          BatchNorm(3), 
          Conv((1, 1), 3 => 3, x -> leakyrelu(x, 0.01)),
          BatchNorm(3)
        ) |> gpu

imgf = channelview(img1) .|> Float32 
imgf = permutedims(imgf, (3,2,1)) |> Flux.unsqueeze(4) |> gpu

o = m(imgf)[:,:,:, 1] |> cpu 
o = permutedims(o, (3, 2, 1))

colorview(RGB, o)






























