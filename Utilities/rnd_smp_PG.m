function XN = rnd_smp_PG(TrainingNoisy, num_patch_N, par)

Nim_path = fullfile(TrainingNoisy,'*.jpg');
Nim_dir = dir(Nim_path);
Nim_num = length(Nim_dir);
% noisy patches per image
nper_img_N = zeros(1, Nim_num);
for ii = 1:Nim_num
    Nim = im2double(imread(fullfile(TrainingNoisy, Nim_dir(ii).name)));
    [h,w,ch] = size(Nim);
    if h >= 1024
        randh = randi(h-1024);
        Nim = Nim(randh+1:randh+1024,:,:);
    end
    if w >= 1024
        randw = randi(w-1024);
        Nim = Nim(:,randw+1:randw+1024,:);
    end
    nper_img_N(ii) = numel(Nim);
end
nper_img_N = floor(nper_img_N*num_patch_N/sum(nper_img_N));
% extract noisy PGs
XN = [];
XNmean = [];
for ii = 1:Nim_num
    patch_num = nper_img_N(ii);
    Nim = im2double(imread(fullfile(TrainingNoisy, Nim_dir(ii).name)));
    [h,w,ch] = size(Nim);
    par.h = h;
    par.w = w;
    par.ch = ch;
    [NPG, NPGmean] = sample_RGB_PGs(Nim, patch_num, par);
    XN = [XN, NPG];
    XNmean = [XNmean, NPGmean];
end
XN = XN - XNmean;
num_patch = size(XN,2);

patch_path = ['Data/rnd_RGB_PG_' num2str(par.patch_size) 'x' num2str(par.patch_size) '_' num2str(num_patch)  '_' datestr(now, 30) '.mat'];
save(patch_path, 'XN');