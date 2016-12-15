% Layer-wise Multi-Layer Coupled Orthogonal Dictionary Learning

clear;
addpath('Data');
addpath('Utilities');

task = 'BID';
load Data/GMM_RGB_PGs_10_6x6_33_20161205T230237.mat;
%% Parameters Setting
% fixed parameters
par.cls_num            =    cls_num;
par.ps                =   6;
par.nIter           =       100;
par.epsilon         =       1e-3;
% par.Layer = 2;
% tunable parameters
for lambda1 = [0.01]
    for lambda2 = [0.001]
        for lambda3 = [0.001]
            for lambda4 = [0.001]
                for lambda5 = [0.001 0.003 0.005 0.01]
                    lambda = [lambda1, lambda2, lambda3,lambda4,lambda5];
                    par.Layer = length(lambda);
                    %% Coupled ODL
                    PSNR = zeros(par.cls_num, par.Layer+1);
                    SSIM  = zeros(par.cls_num, par.Layer+1);
                    for cls = 1 : par.cls_num
                        fprintf('Coupled_ODL_RGB_PG, Cluster: %d\n', cls);
                        XN = double(Xn{cls});
                        XC = double(Xc{cls});
                        if size(XN, 2)>2e4
                            XN = XN(:,1:2e4);
                            XC = XC(:,1:2e4);
                        end
                        PSNR(cls ,1) = csnr( XN*255, XC*255, 0, 0 );
                        SSIM(cls ,1) = cal_ssim( XN*255, XC*255, 0, 0 );
                        fprintf('The initial PSNR = %2.4f, SSIM = %2.4f. \n', PSNR(cls ,1), SSIM(cls ,1) );
                        %% Initialization of Dictionaries and Coefficients
                        [Dn,~,~] = svd(cov(XN'));
                        [Dc,~,~] = svd(cov(XC'));
                        A = [Dn;Dc]' * [XN;XC];
                        for L = 1: par.Layer
                            %% tunable parameters
                            par.lambda1 = lambda(L);
                            %% Orthogonal Dictionary Learning
                            [A, Dc, Dn, f] = Coupled_ODL_RGB_PG_DL(XC, XN, Dc, Dn, A, par);
                            XN = Dn*A;
                            PSNR(cls, L+1) = csnr( XN*255, XC*255, 0, 0 );
                            SSIM(cls, L+1) = cal_ssim( XN*255, XC*255, 0, 0 );
                            fprintf('The %d-th final PSNR = %2.4f, SSIM = %2.4f. \n', L, PSNR(cls ,L+1), SSIM(cls ,L+1) );
                            %% save results
                            CODL.DC{cls,L}  = Dc;
                            CODL.DN{cls,L}  = Dn;
                            CODL.f{cls,L}  = f;
                            Dict_BID = sprintf('Data/Coupled_ODL_LWML_DL_RGB_PG_%s_%2.4f_%2.4f_%2.4f_%2.4f_%2.4f.mat',task,lambda1,lambda2,lambda3,lambda4,lambda5);
                            save(Dict_BID,'CODL', 'PSNR', 'SSIM');
                        end
                    end
                    mPSNR = mean(PSNR);
                    mSSIM = mean(SSIM);
                    fprintf('The %d-th final PSNR = %2.4f, SSIM = %2.4f. \n', L, PSNR(cls ,L+1), SSIM(cls ,L+1) );
                    Dict_BID = sprintf('Data/Coupled_ODL_LWML_DL_RGB_PG_%s_%2.4f_%2.4f_%2.4f_%2.4f_%2.4f.mat',task,lambda1,lambda2,lambda3,lambda4,lambda5);
                    save(Dict_BID,'CODL', 'PSNR', 'SSIM','mPSNR','mSSIM');
                end
            end
        end
    end
end