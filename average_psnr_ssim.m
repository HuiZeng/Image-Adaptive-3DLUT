test_path = 'images/LUTs/paired/fiveK_480p_3LUT_sm_1e-4_mn_10_sRGB_145';
gt_path = '../data/fiveK/expertC/JPG/480p';
path_list = dir(fullfile(test_path,'*.png'));
img_num = length(path_list);
%calculate psnr
total_psnr = 0;
total_ssim = 0;
total_color = 0;
if img_num > 0 
   for j = 1:img_num 
       image_name = path_list(j).name;
       input = imread(fullfile(test_path,image_name));
       gt = imread(fullfile(gt_path,[image_name(1:end-3), 'jpg']));

       psnr_val = psnr(im2double(input), im2double(gt));
       total_psnr = total_psnr + psnr_val;
       
       ssim_val = ssim(input, gt);
       total_ssim = total_ssim + ssim_val;
       
       color = sqrt(sum((rgb2lab(gt) - rgb2lab(input)).^2,3));
       color = mean(color(:));
       total_color = total_color + color;
       fprintf('%d %f %f %f %s\n',j,psnr_val,ssim_val,color,fullfile(test_path,image_name));
   end
end
qm_psnr = total_psnr / img_num;
avg_ssim = total_ssim / img_num;
avg_color = total_color / img_num;
fprintf('The avgerage psnr is: %f', qm_psnr);
fprintf('The avgerage ssim is: %f', avg_ssim);
fprintf('The avgerage lab is: %f', avg_color);
