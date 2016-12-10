function [im_out, par] = Coupled_ODL_RGB_PG_ML_RID(IMin,IM_GT,model,CODL,par,param)
%% modified on 20161208
param.lambda2 = par.lambda2;
%% Initialization
im_out = IMin;
for t = 1 : par.Layer
    param.lambda = par.lambda(t);
    if mod(t -1,2) == 0
        [nDCnlYH,~,~,par] = Image2PGs( im_out, par );
        %% GMM: full posterior calculation
        nPG = size(nDCnlYH,2)/par.nlsp; % number of PGs
        PYZ = zeros(model.nmodels,nPG);
        for i = 1:model.nmodels
            sigma = model.covs(:,:,i);
            [R,~] = chol(sigma);
            Q = R'\nDCnlYH;
            TempPYZ = - sum(log(diag(R))) - dot(Q,Q,1)/2;
            TempPYZ = reshape(TempPYZ,[par.nlsp nPG]);
            PYZ(i,:) = sum(TempPYZ);
        end
        %% find the most likely component for each patch group
        [~,cls_idx] = max(PYZ);
        cls_idx=repmat(cls_idx, [par.nlsp 1]);
        cls_idx = cls_idx(:);
        [idx,  s_idx] = sort(cls_idx);
        idx2 = idx(1:end-1) - idx(2:end);
        seq = find(idx2);
        seg = [0; seq; length(cls_idx)];
    end
    %%  Image to PGs
    [nDCnlXC,blk_arrXC,DCXC,par] = Image2PGs( im_out, par);
    [nDCnlXN,~,~,par] = Image2PGs( IMin, par);
    X_hat = zeros(par.ps^2*par.ch,par.maxr*par.maxc,'double');
    W = zeros(par.ps^2*par.ch,par.maxr*par.maxc,'double');
    for   i  = 1:length(seg)-1
        idx          =   s_idx(seg(i)+1:seg(i+1));
        cls       =   cls_idx(idx(1));
        Xn    = nDCnlXN(:, idx);
        Dc    = CODL.DC{cls,t};
        Dn    = CODL.DN{cls,t};
        %% use noisy data to obtain the coefficients
        %    then reconstruct clean data
        %         A = mexLasso(Xn, Dn, param);
        %% use noisy and latent clean data for the coefficients
        %    then reconstruct clean data
        Xc = nDCnlXC(:, idx);
        A = mexLasso([Xn;Xc], [Dn;Dc], param);
        %% Reconstruction
        Xc = Dc * A;
        nDCnlXC(:, idx) = Xc;
        X_hat(:,blk_arrXC(idx)) = X_hat(:,blk_arrXC(idx)) + nDCnlXC(:, idx) + DCXC(:,idx);
        W(:,blk_arrXC(idx))     = bsxfun(@plus,W(:,blk_arrXC(idx)),ones(par.ps^2*par.ch,length(idx)));
    end
    %% PGs to Image
    [im_out, par]  = PGs2Image(X_hat,W,par);
    par.PSNR(par.IMindex, t) = csnr( im_out*255, IM_GT*255, 0, 0 );
    par.SSIM(par.IMindex, t) = cal_ssim( im_out*255, IM_GT*255, 0, 0 );
    fprintf('nInnerLoop %d: PSNR = %2.4f, SSIM = %2.4f. \n', t, par.PSNR(par.IMindex, t), par.SSIM(par.IMindex, t) );
end