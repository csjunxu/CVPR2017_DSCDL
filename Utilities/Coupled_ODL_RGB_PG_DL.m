% Main Function of Coupled Orthogonal Dictionary Learning
% Input:
% A: Initial coupled sparse coefficient
% Xc, Xn    : Image Data Pairs of two domains
% Dc, Dn    : Initial Dictionaries
% par : Parameters
%
% Output:
% A: Output coupled sparse coefficient
% Dc, Dn : Output Coupled Dictionaries

function [A, Dc, Dn,f] = Coupled_ODL_RGB_PG_DL(Xc, Xn, Dc, Dn, A, par)

%% parameter setting
param.lambda        = 	    par.lambda1; % not more than 20 non-zeros coefficients
% param.lambda2       =       par.lambda2;

f = 0;

%% Iteratively solving:
% coefficients: A
% dictionary: D

for t = 1 : par.nIter
    %% Updating A
    f_prev = f;
    A = mexLasso([Xn; Xc], [Dn; Dc], param);
    
    %% Updating Dn and Dc
    [Un,~,Vn] = svd(Xn*A','econ');
    Dn = Un*Vn';
    [Uc,~,Vc] = svd(Xc*A','econ');
    Dc = Uc*Vc';
    
    %% Find if converge (NEED MODIFICATION)
    P1 = Xc - Dc * A;
    P1 = P1(:)'*P1(:);
    P2 = Xn - Dn * A;
    P2 = P2(:)'*P2(:);
    P3 = par.lambda1 *  norm(A, 1);
    f = P1 + P2 + P3;
    
    %% if converge then break
    if (abs(f_prev - f) / f < par.epsilon)
        break;
    end
    fprintf('Energy %d: %d\n', t, f);
end

