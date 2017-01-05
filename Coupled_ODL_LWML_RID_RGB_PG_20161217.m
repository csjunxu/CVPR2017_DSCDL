clear;
addpath('Data');
addpath('Utilities');
% GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_MeanImage\';
% GT_fpath = fullfile(GT_Original_image_dir, '*.png');
% TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_NoisyImage\';
% TT_fpath = fullfile(TT_Original_image_dir, '*.png');
GT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_ccnoise_denoised_part\';
GT_fpath = fullfile(GT_Original_image_dir, '*mean.png');
TT_Original_image_dir = 'C:\Users\csjunxu\Desktop\CVPR2017\cc_Results\Real_ccnoise_denoised_part\';
TT_fpath = fullfile(TT_Original_image_dir, '*real.png');
GT_im_dir  = dir(GT_fpath);
TT_im_dir  = dir(TT_fpath);
im_num = length(TT_im_dir);

%% load parameters and dictionary
load Data/params.mat par;
load Data/Coupled_ODL_LWML_DL_RGB_PG_BID_0.0100_0.0010_0.0010_0.0010_0.0030.mat CODL;
load Data/GMM_RGB_PGs_10_6x6_33_20161205T230237.mat;
par.cls_num = 31;
par.Layers = 3;

par.Layer = 1;
par.lambda = zeros(par.cls_num,par.Layers);
for testcluster = 1:1:par.cls_num
    par.testcluster = testcluster;
    for lambda = 0.01:0.02:0.05
        par.lambda(testcluster,par.Layer) = lambda;
        CCPSNR = [];
        CCSSIM = [];
        for i = 1:im_num
            IM_GT = im2double(imread(fullfile(GT_Original_image_dir, GT_im_dir(i).name)));
            IMin = im2double(imread(fullfile(TT_Original_image_dir, TT_im_dir(i).name)));
            S = regexp(TT_im_dir(i).name, '\.', 'split');
            IMname = S{1};
            fprintf('%s : \n',IMname);
            CCPSNR = [CCPSNR csnr( IMin*255,IM_GT*255, 0, 0 )];
            CCSSIM = [CCSSIM cal_ssim( IMin*255, IM_GT*255, 0, 0 )];
            fprintf('The initial PSNR = %2.4f, SSIM = %2.4f. \n',CCPSNR(end), CCSSIM(end));
            [h,w,ch] = size(IMin);
            par.IMindex = i;
            [IMout, par] = Coupled_ODL_RGB_PG_ML_RID(IMin,IM_GT,model,CODL,par);
            %% output
            % imwrite(IMout, ['results/DSCDL_' IMname '_' num2str(lambda) '_' num2str(lambda2) '_' num2str(sqrtmu) '.png']);
        end
        PSNR = par.PSNR;
        SSIM = par.SSIM;
        mPSNR = mean(PSNR);
        mSSIM = mean(SSIM);
        savename = ['Real_RID_CODL_LWML_3_1_' num2str(lambda) '.mat'];
        save(savename, 'mPSNR', 'mSSIM', 'PSNR', 'SSIM');
    end
end