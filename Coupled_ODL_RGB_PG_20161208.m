clear;
addpath('Data');
addpath('Utilities');

task = 'BID';
load Data/GMM_RGB_PGs_10_6x6_33_20161205T230237.mat;
%% Parameters Setting
% tunable parameters
par.lambda1         =       0.01;
par.lambda2         =       0.001;
% fixed parameters 
par.cls_num            =    cls_num;
par.ps                =   6;
par.nIter           =       100;
par.epsilon         =       1e-4;
%% Coupled ODL
for i = 1 : par.cls_num
    XN = double(Xn{i});
    XC = double(Xc{i});
    if size(XN, 2)>2e4
        XN = XN(:,1:2e4);
        XC = XC(:,1:2e4);
    end
    fprintf('Coupled_ODL_RGB_PG, Cluster: %d\n', i);
    %% Initialization of Dictionaries and Coefficients
    [Dn,~,~] = svd(cov(XN'));
    [Dc,~,~] = svd(cov(XC'));
    A = [Dn;Dc]' * [XN;XC];
    %% Orthogonal Dictionary Learning
    [A, Dc, Dn, f] = Coupled_ODL(XC, XN, Dc, Dn, A, par);
    CODL.DC{i} = Dc;
    CODL.DN{i} = Dn;
    CODL.f{i} = f;
    Dict_BID = sprintf('Data/Coupled_ODL_RGB_PG_10_6x6_33_%s_20161208.mat',task);
    save(Dict_BID,'CODL');
end