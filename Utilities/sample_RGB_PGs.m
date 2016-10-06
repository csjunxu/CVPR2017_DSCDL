function [PG, PGmean] = sample_RGB_PGs(im, patch_num, par)

if size(im, 3) == 3,
    disp('RGB image sampled!');
else
    disp('Grayscale image sampled!');
end

[h, w, ch] = size(im);
par.h = h;
par.w = w;
par.ch = ch;
par.maxr         =  h - par.ps + 1;
par.maxc         =  w - par.ps + 1;
r         =  1:par.maxr;
par.r         =  [r r(end)+1:par.maxr];
c         =  1:par.maxc;
par.c         =  [c c(end)+1:par.maxc];
% Index image patch
Index     =   (1:par.maxr*par.maxc);
par.Index    =   reshape(Index, par.maxr, par.maxc);

k    =  0;
Patches = zeros(par.ps^2,length(r)*length(c), 'double');
% first j then i : column wise;  first i then j : row wise;
% since N = Npatch(:) is column wise, we employ first j then i.
for l = 1:par.ch
    for j  = 1:par.ps
        for i  = 1:par.ps
            k         =  k+1;
            blk    =  im(r-1+i,c-1+j,l);
            Patches(k,:) =  blk(:)';
        end
    end
end
x = randperm(h-2*par.ps-1) + par.ps;
y = randperm(w-2*par.ps-1) + par.ps;

[X,Y] = meshgrid(x,y);

xrow = X(:);
ycol = Y(:);

im = double(im);
im = double(im);

PG = [];
PGmean = [];
idx=1;ii=1;n=length(xrow);
while (idx < patch_num) && (ii<=n),
    row = xrow(ii);
    col = ycol(ii);
    % get PG from P
    [Patch, PatchGroup, PatchGroupmean] = Get_PGfP(Patches,row,col,par);
    np = Patch(:) - mean(Patch(:));
    npnorm = sqrt(sum(np.^2));
    np_normalised=reshape(np/npnorm,par.ps,par.ps, ch);
    % eliminate that small variance patch
    if var(np)>0.001
        % eliminate stochastic patch
        if dominant_measure(np_normalised)>par.R_thresh
            %if dominant_measure_G(Lpatch1,Lpatch2)>R_thresh
            PG = [PG PatchGroup];
            PGmean = [PGmean PatchGroupmean];
            idx=idx+1;
        end
    end
    ii=ii+1;
end
fprintf('sampled %d patches.\r\n',patch_num);
end

function R = dominant_measure(p)
% calculate the dominant measure
% ref paper: Eigenvalues and condition numbers of random matries, 1988
% p = size n x n patch

hf1 = [-1,0,1];
vf1 = [-1,0,1]';
Gx = convn(p, hf1,'same');
Gy = convn(p, vf1,'same');

G=[Gx(:),Gy(:)];
[U, S, V]=svd(G);

R=(S(1,1)-S(2,2))/(S(1,1)+S(2,2));

end
