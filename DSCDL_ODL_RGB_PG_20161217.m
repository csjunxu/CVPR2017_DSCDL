clear;
addpath('Data');
addpath('Utilities');
addpath('SPAMS');

task = 'BID';
load Data/GMM_RGB_PGs_10_6x6_33_20161006T220226.mat;
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
for i = 1 : par.cls_num
    XN = double(Xn{i});
    XC = double(Xc{i});
    if size(XN, 2)>2e4
        XN = XN(:,1:2e4);
        XC = XC(:,1:2e4);
    end
    fprintf('DSCDL_RGB_PGs: Cluster: %d\n', i);
        %% Initialization of Dictionaries and Coefficients
    [Dn,~,~] = svd(cov(XN'));
    [Dc,~,~] = svd(cov(XC'));
    Alphan = Dn' * XN;
    Alphac = Dc' * XC;
    [Alphac, Alphan, XC, XN, Dc, Dn, Uc, Un, f] = Double_Semi_Coupled_ODL(Alphac, Alphan, XC, XN, Dc, Dn, par);
    DSCDL.DC{i} = Dc;
    DSCDL.DN{i} = Dn;
    DSCDL.UC{i} = Uc;
    DSCDL.UN{i} = Un;
    DSCDL.f{i} = f;
    Dict_BID = sprintf('Data/DSCDL_Dict_RGB_PGs_10_6x6_33_%s_20161006.mat',task);
    save(Dict_BID,'DSCDL');
end