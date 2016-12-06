clear;
addpath('Data');
addpath('Utilities');
addpath('SPAMS');

task = 'BID';
load Data/GMM_RGB_PGs_10_6x6_33_20161205T230237.mat;
% load Data/rnd_RGB_PG_6x6_1922650_20161006T194212.mat;
% Parameters Setting
par.cls_num            =    cls_num;
par.step               =    3;
par.ps                =   6;
par.rho = 5e-2;
par.lambda1         =       0.01;
par.lambda2         =       0.001;
par.mu              =       0.01;
par.sqrtmu          =       sqrt(par.mu);
par.nu              =       0.1;
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
%
Layer = 2;
PSNR = zeros(par.cls_num,Layer+1);
SSIM = zeros(par.cls_num,Layer+1);
for i = 1 : par.cls_num
    XN = double(Xn{i});
    XC = double(Xc{i});
    if size(XN, 2)>2e4
        XN = XN(:,1:2e4);
        XC = XC(:,1:2e4);
    end
    fprintf('DSCDL_RGB_PGs: Cluster: %d\n', i);
    PSNR(i ,1) = csnr( XN*255, XC*255, 0, 0 );
    SSIM(i ,1) = cal_ssim( XN*255, XC*255, 0, 0 );
    fprintf('The %dth initial PSNR = %2.4f, SSIM = %2.4f. \n', L, PSNR(i ,1), SSIM(i ,1) );
    D = mexTrainDL([XN;XC], param);
    Dn = D(1:size(XN,1),:);
    Dc = D(size(XN,1)+1:end,:);
    Alphac = mexLasso([XN;XC], D, param);
    Alphan = Alphac;
    clear D;
    for L = 1:Layer
        [Alphac, Alphan, Dc, Dn, Uc, Un, f] = ADPU_Double_Semi_Coupled_DL(Alphac, Alphan, XC, XN, Dc, Dn, par);
        %%
        XN = Dn*Alphan;
        PSNR(i, L+1) = csnr( XN*255, XC*255, 0, 0 );
        SSIM(i, L+1) = cal_ssim( XN*255, XC*255, 0, 0 );
        fprintf('The %dth final PSNR = %2.4f, SSIM = %2.4f. \n', L, PSNR(i ,L+1), SSIM(i ,L+1) );
        %% save results
        DSCDL.DC{i,L} = Dc;
        DSCDL.DN{i,L} = Dn;
        DSCDL.UC{i,L} = Uc;
        DSCDL.UN{i,L} = Un;
        DSCDL.f{i,L} = f;
        Dict_BID = sprintf('Data/DSCDL_RGB_PGs_ML_DL_10_6x6_33_%s_20161006.mat',task);
        save(Dict_BID,'DSCDL');
    end
end