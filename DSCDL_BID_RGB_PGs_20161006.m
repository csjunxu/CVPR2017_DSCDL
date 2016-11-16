clear;
addpath('Data');
addpath('Utilities');
addpath('SPAMS');
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
load Data/params.mat par param;
load Data/DSCDL_Dict_RGB_PGs_10_6x6x3_31_BID_20161006.mat DSCDL;
load Data/GMM_RGB_PGs_10_6x6x3_31_20161006T220226.mat;
par.cls_num = 31;

for lambda = [0.01 0.001]
    param.lambda = lambda;
    for lambda2 = [0.01 0.001]
        param.lambda2 = lambda2;
        for sqrtmu = 0.01
            par.sqrtmu = sqrtmu;
            for nInnerLoop = 4
                par.nInnerLoop = nInnerLoop;
                PSNR = [];
                SSIM = [];
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
                    %%
                    %                     nOuterLoop = 1;
                    %                     Continue = true;
                    %                     while Continue
                    % fprintf('Iter: %d \n', nOuterLoop);
                    IMout = DSCDL_RGB_PG_BID_20161011(IMin,IM_GT,model,DSCDL,par,param);
                    % Noise Level Estimation
                    %                         nSig = NoiseEstimation(IMout*255,6);
                    %                         fprintf('The noise level is %2.4f.\n',nSig);
                    %                         if nSig < 0.0001 || nOuterLoop >= 5
                    %                             Continue = false;
                    %                         else
                    %                             nOuterLoop = nOuterLoop + 1;
                    %                             IMin = IMout;
                    %                         end
                    %                     end
                    fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', csnr( IMout*255,IM_GT*255, 0, 0 ), cal_ssim( IMout*255, IM_GT*255, 0, 0 ));
                    %% output
%                     imwrite(IMout, ['results/DSCDL_' IMname '_' num2str(lambda) '_' num2str(lambda2) '_' num2str(sqrtmu) '.png']);
                    PSNR =[PSNR csnr( IMout*255,IM_GT*255, 0, 0 )];
                    SSIM = [SSIM cal_ssim( IMout*255, IM_GT*255, 0, 0 )];
                    fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', PSNR(end), SSIM(end));
                end
                mPSNR = mean(PSNR);
                mSSIM = mean(SSIM);
                savename = ['Real_DSCDL_' num2str(lambda) '_' num2str(lambda2) '_' num2str(sqrtmu) '.mat'];
                save(savename, 'mPSNR', 'mSSIM', 'PSNR', 'SSIM');
            end
        end
    end
end