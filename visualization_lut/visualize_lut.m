
LUT1 = [];
LUT_name = ['visualization/learned_LUT_234_1.txt'];
f = fopen(LUT_name,'r');
for i = 1:n^3*3
   LUT1(i) = str2double(fgetl(f)); 
end
fclose(f);

LUT2 = [];
LUT_name = ['visualization/learned_LUT_234_2.txt'];
f = fopen(LUT_name,'r');
for i = 1:n^3*3
   LUT2(i) = str2double(fgetl(f)); 
end
fclose(f);

LUT3 = [];
LUT_name = ['visualization/learned_LUT_234_3.txt'];
f = fopen(LUT_name,'r');
for i = 1:n^3*3
   LUT3(i) = str2double(fgetl(f)); 
end
fclose(f);

r0 = repmat(linspace(0,1,n)',1,n,n);
g0 = repmat(linspace(0,1,n),n,1,n);
b0 = repmat(reshape(linspace(0,1,n),[1,1,n]),n,n);

%adaptive weight 1
% a1 = 2.49;
% a2 = -1.92;
% a3 = -0.33;
a1 = 1.85;
a2 = -0.09;
a3 = -0.91;
LUT = LUT1 * a1 + LUT2 * a2 + LUT3 * a3;
r = LUT(1:n^3);r = reshape(r,[n,n,n]);
g = LUT(n^3+1:n^3*2);g = reshape(g,[n,n,n]);
b = LUT(n^3*2+1:n^3*3);b = reshape(b,[n,n,n]);
C = [r(:),g(:),b(:)];
figure(1);
scatter3(r0(:),g0(:),b0(:),20,C,'filled');

% adaptive weight 2
a1 = 1.59;
a2 = 0.99;
a3 = -1.18;
LUT = LUT1 * a1 + LUT2 * a2 + LUT3 * a3;
r = LUT(1:n^3);r = reshape(r,[n,n,n]);
g = LUT(n^3+1:n^3*2);g = reshape(g,[n,n,n]);
b = LUT(n^3*2+1:n^3*3);b = reshape(b,[n,n,n]);
C = [r(:),g(:),b(:)];
figure(2);
scatter3(r0(:),g0(:),b0(:),20,C,'filled');

% plot used in the paper
n = 33;
fontsize = 12;
set(gca,'FontSize',fontsize)
figure(3);
for i = 1:8:n%[1,17,33]
    r_slice = squeeze(r(i,:,:));
    g_slice = squeeze(g(:,i,:));
    b_slice = squeeze(b(:,:,i));
    r2 = repmat(linspace(0,1,n)',1,n);
    g2 = repmat(linspace(0,1,n),n,1);

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
        
