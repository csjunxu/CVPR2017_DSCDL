clear;clc;
addpath('Data');
addpath('Utilities');
addpath('SPAMS');

load Data/params.mat;
task = 'BID';
load Data/GMM_RGB_PGs_10_6x6_33_20161006T220226.mat;
for i = 1 : par.cls_num
    XN_t = double(Xn{i});
    XC_t = double(Xc{i});
    fprintf('DSCDL:_RGB_PGs: Cluster: %d\n', i);
    D = mexTrainDL([XN_t;XC_t], param);
    Dn = D(1:size(XN_t,1),:);
    Dc = D(size(XN_t,1)+1:end,:);
    Alphac = mexLasso([XN_t;XC_t], D, param);
    Alphan = Alphac;
    clear D;
    [Alphac, Alphan, XC_t, XN_t, Dc, Dn, Uc, Un, Pn, f] = ADPU_Double_Semi_Coupled_DL(Alphac, Alphan, XC_t, XN_t, Dc, Dn, par);
    DSCDL.DC{i} = Dc;
    DSCDL.DN{i} = Dn;
    DSCDL.UC{i} = Uc;
    DSCDL.UN{i} = Un;
    DSCDL.PN{i} = Pn;
    DSCDL.f{i} = f;
    Dict_BID = sprintf('Data/DSCDL_Dict_RGB_PGs_10_6x6_33_%s_20161006.mat',task);
    save(Dict_BID,'DSCDL');
end