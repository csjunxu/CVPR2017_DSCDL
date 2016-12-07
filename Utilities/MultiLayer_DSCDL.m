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

function [DSCDL, par] = MultiLayer_DSCDL(Ac, An, Xc, Xn, Dc, Dn, par)

par.PSNR(par.i ,1) = csnr( Xn*255, Xc*255, 0, 0 );
par.SSIM(par.i ,1) = cal_ssim( Xn*255, Xc*255, 0, 0 );
fprintf('The initial PSNR = %2.4f, SSIM = %2.4f. \n', par.PSNR(par.i ,1), par.SSIM(par.i ,1) );


%% parameter setting
param.lambda        = 	    par.lambda1; % not more than 20 non-zeros coefficients
param.lambda2       =       par.lambda2;
param.mode          = 	    2;       % penalized formulation
param.approx=0;
param.K = par.K;
param.L = par.L;
f = 0;

%% Initialize Us, Up as I
Us = eye(size(Dn, 2));
Up = eye(size(Dn, 2));


for L = 1: par.Layer
    
    %% Iteratively solve A D U
    
    for t = 1 : par.nIter
        
        %% Updating Alphas and Alphap
        f_prev = f;
        An = mexLasso([Xn; par.sqrtmu * Up * full(Ac)], [Dn; par.sqrtmu * Us],param);
        Ac = mexLasso([Xc; par.sqrtmu * Us * full(An)], [Dc; par.sqrtmu * Up],param);
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
        Us = (1 - par.rho) * Us  + par.rho * Up * Ac * An' / ( An * An' + par.nu * eye(size(An, 1)));
        Up = (1 - par.rho) * Up  + par.rho * Us * An * Ac' / ( Ac * Ac' + par.nu * eye(size(Ac, 1)));
        
        %% Find if converge (NEED MODIFICATION)
        P1 = Xc - Dc * Ac;
        P1 = P1(:)'*P1(:) / 2;
        P2 = par.lambda1 *  norm(Ac, 1);
        P3 = Us * An - Up * Ac;
        P3 = P3(:)'*P3(:) / 2;
        P4 = par.nu * norm(Up, 'fro');
        fp = 1 / 2 * P1 + P2 + par.mu * (P3 + P4);
        
        P1 = Xn - Dn * An;
        P1 = P1(:)'*P1(:) / 2;
        P2 = par.lambda1 *  norm(An, 1);
        P3 = Us * An - Up * Ac;
        P3 = P3(:)'*P3(:) / 2;
        P4 = par.nu * norm(Us, 'fro');
        fs = 1 / 2 * P1 + P2 + par.mu * (P3 + P4);
        
        f = fp + fs;
        
        %% if converge then break
        if (abs(f_prev - f) / f < par.epsilon)
            break;
        end
        fprintf('Energy %d: %d\n', t, f);
    end
    %%
    Xn = Dn*An;
    par.PSNR(par.i, L+1) = csnr( Xn*255, Xc*255, 0, 0 );
    par.SSIM(par.i, L+1) = cal_ssim( Xn*255, Xc*255, 0, 0 );
    fprintf('The %dth final PSNR = %2.4f, SSIM = %2.4f. \n', L, par.PSNR(par.i ,L+1), par.SSIM(par.i ,L+1) );
    %% save results
    DSCDL.DC{i,L} = Dc;
    DSCDL.DN{i,L} = Dn;
    DSCDL.UC{i,L} = Uc;
    DSCDL.UN{i,L} = Un;
    DSCDL.f{i,L} = f;
    Dict_BID = sprintf('Data/DSCDL_RID_RGB_PG_ML_DL_10_6x6_33_20161207.mat');
    save(Dict_BID,'DSCDL','par');
end
