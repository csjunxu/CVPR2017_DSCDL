% Main Function of Coupled Dictionary Learning
% Input:
% Alphap,Alphas: Initial sparse coefficient of two domains
% Xp    ,Xs    : Image Data Pairs of two domains
% Dp    ,Ds    : Initial Dictionaries
% Wp    ,Ws    : Initial Projection Matrix
% par          : Parameters
%
% Output
% Alphap,Alphas: Output sparse coefficient of two domains
% Dp    ,Ds    : Output Coupled Dictionaries
% Up    ,Us    : Output Projection Matrix for Alpha
%

function [Ac, An, Dc, Dn, Pc, Pn, f] = Double_Semi_Coupled_ODL(Xc, Xn, Dc, Dn, Ac, An, par)

%% parameter setting
param.lambda        = 	    par.lambda1; % not more than 20 non-zeros coefficients
param.lambda2       =       par.lambda2;
param.mode          = 	    2;       % penalized formulation
param.approx=0;
param.K = par.K;
param.L = par.L;
param.iter=300;

f = 0;

%% Initialize Us, Up as I
Pn = eye(size(Dn, 2));
Pc = eye(size(Dn, 2));

%% Iteratively solving: 
% coefficients: A 
% dictionary: D 
% Linear Mapping: U
%

for t = 1 : par.nIter
    
    %% Updating An and Ac
    f_prev = f;
    An = mexLasso([Xn; par.sqrtmu * Pc * full(Ac)], [Dn; par.sqrtmu * Pn],param);
    Ac = mexLasso([Xc; par.sqrtmu * Pn * full(An)], [Dc; par.sqrtmu * Pc],param);
    
    %% Updating Dn and Dc
    [Un,~,Vn] = svd(Xn*An','econ');
    Dn = Un*Vn';
    [UC,~,VC] = svd(XC*AC','econ');
    Dc = UC*VC';

    %% Updating Pn and Pc
    Pn = (1 - par.rho) * Pn  + par.rho * Pc * Ac * An' / ( An * An' + par.nu * eye(size(An, 1)));
    Pc = (1 - par.rho) * Pc  + par.rho * Pn * An * Ac' / ( Ac * Ac' + par.nu * eye(size(Ac, 1)));
    
    %% Find if converge (NEED MODIFICATION)
    P1 = Xc - Dc * Ac;
    P1 = P1(:)'*P1(:) / 2;
    P2 = par.lambda1 *  norm(Ac, 1);
    P3 = Pn * An - Pc * Ac;
    P3 = P3(:)'*P3(:) / 2;
    P4 = par.nu * norm(Pc, 'fro');
    fp = 1 / 2 * P1 + P2 + par.mu * (P3 + P4);
    
    P1 = Xn - Dn * An;
    P1 = P1(:)'*P1(:) / 2;
    P2 = par.lambda1 *  norm(An, 1);
    P3 = Pn * An - Pc * Ac;
    P3 = P3(:)'*P3(:) / 2;
    P4 = par.nu * norm(Pn, 'fro');
    fs = 1 / 2 * P1 + P2 + par.mu * (P3 + P4);
    
    f = fp + fs;
    
    %% if converge then break
    if (abs(f_prev - f) / f < par.epsilon)
        break;
    end
    fprintf('Energy %d: %d\n', t, f);
end

