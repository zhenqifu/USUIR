Files = dir(strcat('../J/','*.jpg'));
number=length(Files);
for i=1:number
    i
    filename = Files(i).name;
    img = imread(['../J/',filename]);
    uciqe = UCIQE(img);
    uciqe = roundn(uciqe,-4);
    score(i) = uciqe;
end

mean(score)
