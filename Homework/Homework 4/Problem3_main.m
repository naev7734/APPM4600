clc
clear all
close all

x0 = [0;2;3];
tol = 10^(-6);
nmax = 1000;
verb = 1;
x_exact = [0;.1;1];

fprintf('Newton Method')
[r,rn] = newton_method_nd(@fun,@Jfun,x0,tol,nmax,verb);
for i = 1:length(rn)
    norm_x_Newton(1,i) = norm(rn(:,i));
    err_x_Newton(1,i) = norm(rn(:,i)-x_exact);
end


fprintf('\nSteepest Descent')
%type = 'armijo';
type = 'swolfe';
[r,rn] = steepest_descent(@fun,@qfun,x0,tol,nmax,type,verb);

for i = 1:length(rn)
    norm_x_steep(1,i) = norm(rn(:,i));
    err_x_steep(1,i) = norm(rn(:,i)-x_exact);
end


fprintf('\nCombination')
tol_1 = 5*10^(-2);
[r,rn] = steepest_descent(@fun,@qfun,x0,tol_1,nmax,type,verb);
for i = 1:length(rn)
    norm_x_comb(1,i) = norm(rn(:,i));
    err_x_comb(1,i) = norm(rn(:,i)-x_exact);
end

tol_2 = 10^(-6);
[r,rn] = newton_method_nd(@fun,@Jfun,r,tol,nmax,verb);

for k = 1:length(rn)
    norm_x_comb(1,i+k) = norm(rn(:,k));
    err_x_comb(1,i+k) = norm(rn(:,k)-x_exact);
end

figure(1)
semilogy(1:length(norm_x_Newton),norm_x_Newton,1:length(norm_x_steep),norm_x_steep,1:length(norm_x_comb),norm_x_comb)
title('Convergence of Different Methods')
legend('Newton Method','Steepest Descent','Combination')
xlabel('Number of Iterations')
ylabel('Norm of Solution')
set(gca, 'yscale', 'log')

figure(2)
plot(1:length(err_x_Newton),err_x_Newton,1:length(err_x_steep),err_x_steep,1:length(err_x_comb),err_x_comb)
%semilogy(1:length(err_x_Newton),norm_x_Newton)
set(gca, 'yscale', 'log')
xlabel('Number of Iterations')
title('Error over Iterations for Differnet Methods')
ylabel('Norm of the Error in the Solution')
legend('Newton Method','Steepest Descent','Combination')



function y = fun(x)
    y(1,1) = x(1) + cos(x(1)*x(2)*x(3)) - 1;
    y(2,1) = (1-x(1))^.25 + x(2) + .05*x(3)^2 - .15*x(3) - 1;
    y(3,1) = -1*x(1)^2 - .1*x(2)^2 + .01*x(2) + x(3) - 1;
end

function J = Jfun(x)
    J(1,1) = 1 - x(2)*x(3)*sin(x(1)*x(2)*x(3));
    J(1,2) = -1*x(1)*x(3)*sin(x(1)*x(2)*x(3));
    J(1,3) = -1*x(1)*x(2)*sin(x(1)*x(2)*x(3));
    J(2,1) = -.25*(1-x(1))^(-.75);
    J(2,2) = 1;
    J(2,3) = .1*x(3) - .15;
    J(3,1) = -2*x(1);
    J(3,2) = -.2*x(2) + .01;
    J(3,3) = 1;
end

% function y = fun(x)
%     y(1,1) = x(1)+x(2)+1;
%     y(2,1) = 2*x(1)*x(2);
% end
% 
% function J = Jfun(x)
%     J(1,1) = 1;
%     J(1,2) = 1;
%     J(2,1) = 2;
%     y(2,2) = 1;
% end

function q = qfun(x)
    J = Jfun(x);
    y = fun(x);
    q = transpose(J)*y;
end

% function q = qfun(x)
%     y = fun(x);
%     q = .5 * y(1)^2 + y(2)^2 + y(3)^2;
% end