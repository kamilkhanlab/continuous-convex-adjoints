%% -----------------------------------------------------------------
 % Last Modified by Yulan Zhang
 % 2024/04/11
 %
 % This script plots the subtangent lines obtained from C++ implementation of adjoint subgradients
 % evalaution system.
 %
 % Example 5 for CACE paper
 % 
%% -----------------------------------------------------------------
% Import data generated in C++
cv = importdata('data_x2cv.txt');
cc = importdata('data_x2cc.txt');
x = importdata('data_x2.txt');
p = linspace(4.28,14.28,length(cv));
p_i = 10.78;
n_p = find(abs(p-p_i) < 1e-12);
xcci = cc(n_p);
xcvi = cv(n_p);

figure(1)
hold on;
plot(p,cv,'r', 'LineWidth', 2.0);
plot(p,cc,'r', 'LineWidth', 2.0);
plot(p,x,'black:', 'LineWidth', 2.0);
plot(p_i, xcvi,'.b','MarkerSize',20)
plot(p_i, xcci, '.b','MarkerSize',20)

%forward ODE system
fplot(@(z) (-3.7132e-03 *(z-p(n_p))+ cv(n_p)),[p(n_p)-1.0,p(n_p)+1.0],'b','LineWidth',1.8 )
fplot(@(z) (-5.6709e-03 *(z-p(n_p))+ cc(n_p)),[p(n_p)-1.0,p(n_p)+1.0],'b', 'LineWidth',1.8 )

%adjoint ODE system with forward-mode AD
fplot(@(z) (-3.7132e-03*(z-p(n_p))+ cv(n_p)),[p(n_p)-0.8,p(n_p)+0.8],'g', 'LineWidth',1.8 )
fplot(@(z) (-5.6709e-03*(z-p(n_p))+ cc(n_p)),[p(n_p)-0.8,p(n_p)+0.8],'g', 'LineWidth',1.8 )

%adjoint ODE system with reverse-mode AD
fplot(@(z) (-3.7051e-03*(z-p(n_p))+ cv(n_p)),[p(n_p)-0.6,p(n_p)+0.6],'k', 'LineWidth',1.8 )
fplot(@(z) (-5.6526e-03*(z-p(n_p))+ cc(n_p)),[p(n_p)-0.6,p(n_p)+0.6],'k', 'LineWidth',1.8 )

grid
xlim([4.28,14.28])
xlabel('p7');
ylabel('x2');
set(gca,'linewidth',1.3)
set(gca,'FontSize',15)
box on 
hold off

