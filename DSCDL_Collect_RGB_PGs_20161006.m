addpath('Data');
addpath('Utilities');
TrainingNoisy = '../TrainingData/RGBNoisy/';

load Data/params.mat;
load ../CVPR2017_PGPD_BID/PG-GMM_TrainingCode/PGGMM_RGB_6x6_3_win15_nlsp10_delta0.002_cls33.mat;
% Parameters Setting
par.ps = ps;
par.cls_num = cls_num;
par.nlsp = nlsp;
par.Win = win;   % size of window around the patch
num_patch_N = 200000;
par.R_thresh = 0.05;

XN = rnd_smp_PG(TrainingNoisy, num_patch_N, par);

%% GMM: full posterior calculation
nPG = size(XN,2)/par.nlsp; % number of PGs
PYZ = zeros(model.nmodels,nPG);
for i = 1:model.nmodels
    sigma = model.covs(:,:,i);
    [R,~] = chol(sigma);
    Q = R'\XN;
    TempPYZ = - sum(log(diag(R))) - dot(Q,Q,1)/2;
    TempPYZ = reshape(TempPYZ,[par.nlsp nPG]);
    PYZ(i,:) = sum(TempPYZ);
end
%% find the most likely component for each patch group
[~,cls_idx] = max(PYZ);
cls_idx=repmat(cls_idx, [par.nlsp 1]);
cls_idx = cls_idx(:);
[idx,  s_idx] = sort(cls_idx);
idx2 = idx(1:end-1) - idx(2:end);
seq = find(idx2);
seg = [0; seq; length(cls_idx)];

load ../CVPR2017_PGPD_BID/PG-GMM_TrainingCode/Kodak24_PGs_6x6_3_10_33.mat;
for   i = 1:length(seg)-1
    idx    =   s_idx(seg(i)+1:seg(i+1));
    cls =   cls_idx(idx(1));
    % given noisy patches, search corresponding clean ones via k-NN
    NPG = XN(:,idx);
    CPG = Xc{cls};
    PGIDX = knnsearch(NPG', CPG');
    Xn{cls} = XN(:, idx);
    Xc{cls} = CPG(:,PGIDX);
end
% save model
GMM_model = ['Data/GMM_RGB_PGs_' num2str(par.nlsp) '_' num2str(par.ps) 'x' num2str(par.ps) '_' num2str(cls_num) '_' datestr(now, 30) '.mat'];
save(GMM_model, 'model', 'Xn','Xc','cls_num');