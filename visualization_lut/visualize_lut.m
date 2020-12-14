
n = 33;
% r = repmat(linspace(0,1,n)',1,n,n);
% g = repmat(linspace(0,1,n),n,1,n);
% b = repmat(reshape(linspace(0,1,n),[1,1,n]),n,n);
% lut = [r(:),g(:),b(:)];

% f = fopen('IdentityLUT33.txt','w');
% for i = 1:length(b(:))
%    fprintf(f,'%.6f  %.6f  %.6f\n',lut(i,1),lut(i,2),lut(i,3)); 
% end
% fclose(f);


% for i = 1:5:n
%     subplot(1,6,1)
%     r_slice = squeeze(r(i,:,:));
%     g_slice = squeeze(g(:,i,:));
%     b_slice = squeeze(b(:,:,i));
%     r2 = repmat(linspace(0,1,n)',1,n);
%     g2 = repmat(linspace(0,1,n),n,1);
% %     plot3(r_slice(:),g_slice(:),b_slice(:));
% %     view(0,90)
%     plot3(g2(:),r2(:),r_slice(:));
%     hold on;
%     view(90,0)
%     zlim([-0.4,1]);
% end

fontsize = 12;
set(gca,'FontSize',fontsize)

LUT1 = [];
LUT_name = ['learned_LUT_234_1.txt'];
f = fopen(LUT_name,'r');
for i = 1:n^3*3
   LUT1(i) = str2double(fgetl(f)); 
end
fclose(f);

LUT2 = [];
LUT_name = ['learned_LUT_234_2.txt'];
f = fopen(LUT_name,'r');
for i = 1:n^3*3
   LUT2(i) = str2double(fgetl(f)); 
end
fclose(f);

LUT3 = [];
LUT_name = ['learned_LUT_234_3.txt'];
f = fopen(LUT_name,'r');
for i = 1:n^3*3
   LUT3(i) = str2double(fgetl(f)); 
end
fclose(f);

a1 = 2.49;
a2 = -1.92;
a3 = -0.33;

a1 = 1.85;
a2 = -0.09;
a3 = -0.91;
% 
a1 = 1.59;
a2 = 0.99;
a3 = -1.18;
LUT = LUT1 * a1 + LUT2 * a2 + LUT3 * a3;


r = LUT(1:n^3);r = reshape(r,[n,n,n]);
g = LUT(n^3+1:n^3*2);g = reshape(g,[n,n,n]);
b = LUT(n^3*2+1:n^3*3);b = reshape(b,[n,n,n]);
%     hist(r(:));
figure(3);
for i = 1:8:n%[1,17,33]
    r_slice = squeeze(r(i,:,:));
    g_slice = squeeze(g(:,i,:));
    b_slice = squeeze(b(:,:,i));
    r2 = repmat(linspace(0,1,n)',1,n);
    g2 = repmat(linspace(0,1,n),n,1);

%         plot3(r_slice(:),g_slice(:),b_slice(:));
%         plot3(g2(:),r2(:),g_slice(:));hold on;
%         surface(g2,r2,g_slice);hold on;view(3)

%         subplot(1,3,k)
%         surface(g2,r2,r_slice);view(3)

    subplot(1,3,1)
    surface(g2,r2,r_slice);view(3)
    set(gca,'FontSize',fontsize)
%     view(90,0)
%     zlim([-0.3,0.9]);

    subplot(1,3,2)
    surface(g2,r2,g_slice);view(3)
    set(gca,'FontSize',fontsize)
%     view(90,0)
%     zlim([-0.3,0.9]);

    subplot(1,3,3)
    surface(g2,r2,b_slice);view(3)
    set(gca,'FontSize',fontsize)
%     view(90,0)
%     zlim([-0.3,0.9]);
end
        