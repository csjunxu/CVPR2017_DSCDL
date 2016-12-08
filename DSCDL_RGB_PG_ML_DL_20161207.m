clear;
addpath('Data');
addpath('Utilities');

task = 'RID';
load Data/GMM_RGB_PGs_10_6x6_33_20161205T230237.mat;
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
%% fixed parameter setting
param.mode          = 	    2;       % penalized formulation
param.approx=0;
param.K = par.K;
param.L = par.L;
param.iter=300;
save Data/MultiLayer_Param_20161207_1.mat par param;

%% begin dictionary learning
PSNR = zeros(par.cls_num, par.Layer+1);
SSIM = zeros(par.cls_num, par.Layer+1);
for i = 1 : par.cls_num
    fprintf('DSCDL_RGB_PG_ML_DL, Cluster: %d\n', i);
    XN = double(Xn{i});
    XC = double(Xc{i});
    if size(XN, 2)>2e4
        XN = XN(:,1:2e4);
        XC = XC(:,1:2e4);
    end
    PSNR(i ,1) = csnr( XN*255, XC*255, 0, 0 );
    SSIM(i ,1) = cal_ssim( XN*255, XC*255, 0, 0 );
    fprintf('The initial PSNR = %2.4f, SSIM = %2.4f. \n', PSNR(i ,1), SSIM(i ,1) );
    
    for L = 1: par.Layer
        %% tunable parameters
        param.lambda        = 	    par.lambda1(L);
        param.lambda2       =       par.lambda2;
        D = mexTrainDL([XN;XC], param);
        Dn = D(1:size(XN,1),:);
        Dc = D(size(XN,1)+1:end,:);
        Ac = mexLasso([XN;XC], D, param);
        An = Ac;
        clear D;
        [Dc, Dn, Uc, Un, Ac, An] = MultiLayer_DSCDL(Ac, An, XC, XN, Dc, Dn, par, param);
        %%
        Xn = Dn*An;
        par.PSNR(i, L+1) = csnr( Xn*255, Xc*255, 0, 0 );
        par.SSIM(i, L+1) = cal_ssim( Xn*255, Xc*255, 0, 0 );
        fprintf('The %d-th final PSNR = %2.4f, SSIM = %2.4f. \n', L, par.PSNR(i ,L+1), par.SSIM(i ,L+1) );
        %% save results
        DSCDL.DC{i,L} = Dc;
        DSCDL.DN{i,L} = Dn;
        DSCDL.UC{i,L} = Uc;
        DSCDL.UN{i,L} = Un;
        DSCDL.f{i,L} = f;
        Dict_BID = sprintf('Data/DSCDL_RID_RGB_PG_ML_DL_10_6x6_33_%2.4f_%2.4f.mat',par.lambda1(1),par.lambda1(2));
        save(Dict_BID,'DSCDL','par');
    end
end