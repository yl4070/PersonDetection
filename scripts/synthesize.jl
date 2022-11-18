


person = load(randimg(1)[1])

sz = size(person) .÷ 3
imresize(person, sz...)


person
img1 + imresize(padarray(person, Fill(0, (500, 700), (500, 700))), size(img1)) 







