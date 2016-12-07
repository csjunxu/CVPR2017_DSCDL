clear;
addpath('Data');
addpath('Utilities');

task = 'BID';
load Data/GMM_RGB_PGs_10_6x6_33_20161205T230237.mat;
% load Data/rnd_RGB_PG_6x6_1922650_20161006T194212.mat;
%% Parameters Setting
% lambda is important;
% lambda2 is not important
% sqrtmu  is not important

% the number of layers in DL
par.Layer = 2;
% tunable parameters
par.lambda1         =       [0.01 0.1];
par.lambda2         =       0.001;

% temporally fixed parameters
par.mu              =       0.01;
par.sqrtmu          =       sqrt(par.mu);
par.nu              =       0.1;

% fully fixed parameters
par.cls_num            =    cls_num;
par.step               =    3;
par.ps                =   6;
par.rho = 5e-2;
par.nIter           =       100;
par.epsilon         =       5e-3;
par.t0              =       5;
par.K               =       256;
par.L               =       par.ps^2;
param.K = par.K;
param.lambda = par.lambda1;
param.lambda2 = par.lambda2;
param.iter=300;
param.L = par.ps^2;


%% begin dictionary learning
par.PSNR = zeros(par.cls_num, par.Layer+1);
par.SSIM = zeros(par.cls_num, par.Layer+1);
load Data/DSCDL_RGB_PGs_ML_DL_10_6x6_33_BID_20161006.mat
for i = 1 : par.cls_num
    XN = double(Xn{i});
    XC = double(Xc{i});
    if size(XN, 2)>2e4
        XN = XN(:,1:2e4);
        XC = XC(:,1:2e4);
    end
    par.i = i;
    fprintf('DSCDL_RGB_PG_ML_DL, Cluster: %d\n', i);
    D = mexTrainDL([XN;XC], param);
    Dn = D(1:size(XN,1),:);
    Dc = D(size(XN,1)+1:end,:);
    Ac = mexLasso([XN;XC], D, param);
    An = Ac;
    clear D;
    [DSCDL, par] = MultiLayer_DSCDL(Ac, An, XC, XN, Dc, Dn, par);
    Dict_BID = sprintf('Data/DSCDL_RID_RGB_PG_ML_DL_10_6x6_33_%2.4f_%2.4f.mat',par.lambda1(1),par.lambda1(2));
    save(Dict_BID,'DSCDL','par');
end