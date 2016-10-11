% Parameters Setting
par.cls_num            =    31;
par.step               =    3;
par.ps                =   6;
par.nlsp = 10;
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

save Data/params.mat par param;