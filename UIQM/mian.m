Files = dir(strcat('../J/','*.jpg'));
number=length(Files);
for i=1:number
    i
    filename = Files(i).name;
    img = imread(['../J/',filename]);
    uiqm = UIQM(img);
    uiqm =roundn(uiqm,-4);
    score(i) = uiqm;
end

mean(score)
