%% -----------------------------------------------------------------
 % Last Modified by Yulan Zhang
 % 2022/08/10
 %
 % This script plots the subtangent lines obtained from C++ implementation of adjoint subgradients
 % evalaution system.
 %
 % Example 3 for CACE paper
 % 
%% -----------------------------------------------------------------
% Import data generated obtained in C++
cv = importdata('data_x2cv.txt');
cc = importdata('data_x2cc.txt');
x = importdata('data_x2.txt');
p = linspace(-6.5,6.5,length(cv));

p_i1 = 5.0;
n_p1 = find(abs(p-p_i1) < 1e-12);
xcci1 = cc(n_p1);
xcvi1= cv(n_p1);

p_i2 = -3.0;
n_p2 = find(abs(p-p_i2) < 1e-12);
xcci2 = cc(n_p2);
xcvi2= cv(n_p2);


% Compute the subtangent line at p1 = -3.0, p2 = 0.5 using finite difference approximation
dP = gradient(p);
dxcv = gradient(cv);
xcv1 = interp1(p, cv, p_i1);
dP1 = interp1(p, dP, p_i1);
dxcv1 = interp1(p, dxcv, p_i1);
Slope_cv_1 = dxcv1/dP1;
YIntercept_cv_1 = xcv1 - Slope_cv_1 * p_i1;
XIntercept_cv_1 = -YIntercept_cv_1 / Slope_cv_1;

dxcc = gradient(cc);
xcc1 = interp1(p, cc, p_i1);
dP1 = interp1(p, dP, p_i1);
dxcc1 = interp1(p, dxcc, p_i1);
Slope_cc_1 = dxcc1/dP1;
YIntercept_cc_1 = xcc1 - Slope_cc_1 * p_i1;
XIntercept_cc_1 = -YIntercept_cc_1 / Slope_cc_1;

%Compute the subtangent line at p1 = 5.0, p2 = 0.5 using finite difference approximation
dP = gradient(p);
dxcv = gradient(cv);
xcv2 = interp1(p, cv, p_i2);
dP2 = interp1(p, dP, p_i2);
dxcv2 = interp1(p, dxcv, p_i2);
Slope_cv_2 = dxcv2/dP2;
YIntercept_cv_2 = xcv2 - Slope_cv_2 * p_i2;
XIntercept_cv_2 = -YIntercept_cv_2 / Slope_cv_2;

dxcc = gradient(cc);
xcc2 = interp1(p, cc, p_i2);
dP2 = interp1(p, dP, p_i2);
dxcc2 = interp1(p, dxcc, p_i2);
Slope_cc_2 = dxcc2/dP2;
YIntercept_cc_2 = xcc2 - Slope_cc_2 * p_i2;
XIntercept_cc_2 = -YIntercept_cc_2 / Slope_cc_2;


% Plot
figure(1)
hold on;
plot(p,cv,'-r', 'LineWidth', 2.0);
plot(p,cc,'-r', 'LineWidth', 2.0);
plot(p,x,'black:', 'LineWidth', 2.0);
plot(p_i1, xcvi1,'.b','MarkerSize',20)
plot(p_i1, xcci1, '.b','MarkerSize',20)
plot(p_i2, xcvi2,'.b','MarkerSize',20)
plot(p_i2, xcci2, '.b','MarkerSize',20)

len1 = 1.0;
len2 = 2.0;

fplot(@(z) (1.3377e-01*(z-p(n_p1))+ cv(n_p1)),[p(n_p1)-len1,p(n_p1)+len1],'b', 'LineWidth',2.5)
fplot(@(z) (-7.8097e-02*(z-p(n_p1))+ cc(n_p1)),[p(n_p1)-len1,p(n_p1)+len1],'b', 'LineWidth',2.5)
fplot(@(z) ( Slope_cv_1*(z-p(n_p1))+ cv(n_p1)),[p(n_p1)-len2,p(n_p1)+len2],':', 'color', '#EDB120','LineWidth', 2.0)
fplot(@(z) ( Slope_cc_1*(z-p(n_p1))+ cc(n_p1)),[p(n_p1)-len2,p(n_p1)+len2],':', 'color', '#EDB120','LineWidth', 2.0)


fplot(@(z) ( 2.6529e-02*(z-p(n_p2))+ cv(n_p2)),[p(n_p2)-len1,p(n_p2)+len1],'b', 'LineWidth',2.5)
fplot(@(z) ( 1.4846e-01*(z-p(n_p2))+ cc(n_p2)),[p(n_p2)-len1,p(n_p2)+len1],'b', 'LineWidth',2.5)
fplot(@(z) ( Slope_cv_2*(z-p(n_p2))+ cv(n_p2)),[p(n_p2)-len2,p(n_p2)+len2],':', 'color', '#EDB120','LineWidth', 2.0)
fplot(@(z) ( Slope_cc_2*(z-p(n_p2))+ cc(n_p2)),[p(n_p2)-len2,p(n_p2)+len2],':', 'color', '#EDB120','LineWidth', 2.0)

grid
xlim([-6.5,6.5])
xlabel('p_1');
ylabel('x_2');
set(gca,'linewidth',1.3)
set(gca,'FontSize',15)
box on 
hold off

