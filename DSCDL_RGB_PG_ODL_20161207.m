clear;
addpath('Data');
addpath('Utilities');

task = 'BID';
load Data/GMM_RGB_PGs_10_6x6_33_20161205T230237.mat;
%% Parameters Setting
% tunable parameters
par.lambda1         =       0.01;
par.lambda2         =       0.001;
par.mu              =       0.01;
par.sqrtmu          =       sqrt(par.mu);
par.nu              =       0.1;
% fixed parameters 
par.cls_num            =    cls_num;
par.step               =    3;
par.ps                =   6;
par.rho = 5e-2;
par.nIter           =       100;
par.epsilon         =       5e-3;
par.t0              =       5;
par.K               =       256;
par.L               =       par.ps^2;
for i = 1 : par.cls_num
    XN = double(Xn{i});
    XC = double(Xc{i});
    if size(XN, 2)>2e4
        XN = XN(:,1:2e4);
        XC = XC(:,1:2e4);
    end
    fprintf('DSCDL_RGB_PG_ODL: Cluster: %d\n', i);
    %% Initialization of Dictionaries and Coefficients
    [Dn,~,~] = svd(cov(XN'));
    [Dc,~,~] = svd(cov(XC'));
    An = Dn' * XN;
    Ac = Dc' * XC;
    %% Orthogonal Dictionary Learning
    [Ac, An, Dc, Dn, Pc, Pn, f] = Double_Semi_Coupled_ODL(XC, XN, Dc, Dn, Ac, An, par);
    DSCDL.DC{i} = Dc;
    DSCDL.DN{i} = Dn;
    DSCDL.PC{i} = Pc;
    DSCDL.PN{i} = Pn;
    DSCDL.f{i} = f;
    Dict_BID = sprintf('Data/DSCDL_Dict_RGB_PGs_10_6x6_33_%s_20161006.mat',task);
    save(Dict_BID,'DSCDL');
end