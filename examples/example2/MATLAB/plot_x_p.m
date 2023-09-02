%% -----------------------------------------------------------------
 % Last Modified by Yulan Zhang
 % 2022/08/10
 % 
 % This script plots the subtangent lines obtained from C++ implementation of adjoint subgradients
 % evalaution system.
 %
 % Example 4 for CACE paper
 %
%% -----------------------------------------------------------------
% Import data generated obtained in C++
cv = importdata('data_xcv.txt');
cc = importdata('data_xcc.txt');
x = importdata('data_x.txt');
p = linspace(-2,2,length(cv));

p_i1 = -1.0;
n_p = find(abs(p-p_i1) < 1e-12);
xcci1 = cc(n_p);
xcvi1 = cv(n_p);


% Compute the subtangent line at p = -1.0 using finite difference approximation
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

% Plot
figure(1)
hold on;
plot(p,cv,'-r', 'LineWidth', 2.0);
plot(p,cc,'-r', 'LineWidth', 2.0);
plot(p,x,'black:', 'LineWidth', 2.0);
plot(p_i1, xcvi1,'.b','MarkerSize',20)
plot(p_i1, xcci1, '.b','MarkerSize',20)


len1 = 0.5;
len2 = 3.0;

fplot(@(z) (6.6153e-01*(z-p(n_p))+ cv(n_p)),[p(n_p)-len1,p(n_p)+len1],'b', 'LineWidth',2.5)
fplot(@(z) (1.2443*(z-p(n_p))+ cc(n_p)),[p(n_p)-len1,p(n_p)+len1],'b', 'LineWidth',2.5)
fplot(@(z) ( Slope_cv_1*(z-p(n_p))+ cv(n_p)),[p(n_p)-len2,p(n_p)+len2],':', 'color', '#EDB120','LineWidth', 2.0)
fplot(@(z) ( Slope_cc_1*(z-p(n_p))+ cc(n_p)),[p(n_p)-len2,p(n_p)+len2],':', 'color', '#EDB120','LineWidth', 2.0)

grid
xlim([-2.0,2.0])
xlabel('p');
ylabel('x');
set(gca,'linewidth',1.3)
set(gca,'FontSize',15)
box on 
hold off

