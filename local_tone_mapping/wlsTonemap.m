
%WLSTONEMAP High Dynamic Range tonemapping using WLS
%
%   The script reduces the dynamic range of an HDR image using the method
%   originally proposed by Durand and Dorsey, "Fast Bilateral Filtering
%   for the Display of High-Dynamic-Range Images",
%   ACM Transactions on Graphics, 2002.
%
%   Instead of the bilateral filter, the edge-preserving smoothing here
%   is based on the weighted least squares(WLS) optimization framework,
%   as described in Farbman, Fattal, Lischinski, and Szeliski,
%   "Edge-Preserving Decompositions for Multi-Scale Tone and Detail
%   Manipulation", ACM Transactions on Graphics, 27(3), August 2008.  


%% Load HDR image from file and convert to greyscale
% hdr = double(hdrread('smallOffice.hdr'));
hdr = double(imread('a1509.jpg'))/255.0;
% hdr = imresize(hdr,0.2);
% hsv = rgb2hsv(hdr);
% I = hsv(:,:,3);
I = 0.2989*hdr(:,:,1) + 0.587*hdr(:,:,2) + 0.114*hdr(:,:,3);
logI = log(I+eps);

%% Perform edge-preserving smoothing using WLS
lambda = 20.0;
alpha  =  1.2;
% base = log(wlsFilter(I, lambda, alpha));
base = log(imguidedfilter(I));

%% Compress the base layer and restore detail
compression = 0.6;
detail = logI - base;
OUT = base*compression + detail;
OUT = exp(OUT);

%% Restore color
OUT = OUT./I;
OUT = hdr .* padarray(OUT, [0 0 2], 'circular' , 'post');
% hsv(:,:,3) = I;
% OUT = hsv2rgb(hsv);

%% Finally, shift, scale, and gamma correct the result
gamma = 1.0/1.0;
bias = -min(OUT(:));
gain = 0.8;
OUT = (gain*(OUT + bias)).^gamma;
% figure
imshow(OUT);
imwrite(OUT,'a1509_T.jpg');
% imshowpair(hdr,OUT, 'montage')
% imshow(hdr,OUT);
