
# utilities for getting image path
imgdir() = joinpath(projectdir(), "imgs")
imgdir(n) = rand(readdir(imgdir()), n)
imgdirabs(n) = map(imgdir(n)) do im
    joinpath(imgdir(), im) 
end
function randimg(n) 
    dir = imgdirabs(1)[1]
    map(rand(readdir(dir), n)) do im
        joinpath(dir, im)
    end
end

begin
    img = randimg(1) |> first |> load
    pd = ceil.((maximum( size(img) ) .- size(img)) ./ 2) .|> Int
    img_ = padarray(img, Pad(:replicate, pd...)) 
    img2 = imresize(img_, (32,32))
    imgf = Float32.(channelview(img2))
    v = var(imgf) 
    @show v
    img2
end












