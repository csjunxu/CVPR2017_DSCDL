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

function [Dc, Dn, Uc, Un, Ac, An] = MultiLayer_DSCDL(Ac, An, Xc, Xn, Dc, Dn, par, param)

f = 0;

%% Initialize Us, Up as I
Un = eye(size(Dn, 2));
Uc = eye(size(Dn, 2));

%% Iteratively solve A D U
for t = 1 : par.nIter
    
    %% Updating Alphas and Alphap
    f_prev = f;
    An = mexLasso([Xn; par.sqrtmu * Uc * full(Ac)], [Dn; par.sqrtmu * Un],param);
    Ac = mexLasso([Xc; par.sqrtmu * Un * full(An)], [Dc; par.sqrtmu * Uc],param);
    dictSize = par.K;
    
    %% Updating Ds and Dp
    for i=1:dictSize
        ai        =    An(i,:);
        Y         =    Xn - Dn * An + Dn(:,i) * ai;
        di        =    Y * ai';
        di        =    di ./ (norm(di,2) + eps);
        Dn(:,i)   =    di;
    end
    
    for i=1:dictSize
        ai        =    Ac(i,:);
        Y         =    Xc - Dc * Ac + Dc(:,i) * ai;
        di        =    Y * ai';
        di        =    di ./ (norm(di,2) + eps);
        Dc(:,i)  =    di;
    end
    %% Updating Ws and Wp => Updating Us and Up
    Un = (1 - par.rho) * Un  + par.rho * Uc * Ac * An' / ( An * An' + par.nu * eye(size(An, 1)));
    Uc = (1 - par.rho) * Uc  + par.rho * Un * An * Ac' / ( Ac * Ac' + par.nu * eye(size(Ac, 1)));
    
    %% Find if converge (NEED MODIFICATION)
    P1 = Xc - Dc * Ac;
    P1 = P1(:)'*P1(:) / 2;
    P2 = param.lambda *  norm(Ac, 1);
    P3 = Un * An - Uc * Ac;
    P3 = P3(:)'*P3(:) / 2;
    P4 = par.nu * norm(Uc, 'fro');
    fp = 1 / 2 * P1 + P2 + par.mu * (P3 + P4);
    
    P1 = Xn - Dn * An;
    P1 = P1(:)'*P1(:) / 2;
    P2 = param.lambda *  norm(An, 1);
    P3 = Un * An - Uc * Ac;
    P3 = P3(:)'*P3(:) / 2;
    P4 = par.nu * norm(Un, 'fro');
    fs = 1 / 2 * P1 + P2 + par.mu * (P3 + P4);
    
    f = fp + fs;
    
    %% if converge then break
    if (abs(f_prev - f) / f < par.epsilon)
        break;
    end
    fprintf('Energy %d: %d\n', t, f);
end
