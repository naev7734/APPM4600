clc
clear all
close all

x0 = [0;0];
tol = 10^(-8);
nmax = 1000;
verb = 1;


fprintf('Newton Method')
    [r,rn] = newton_method_nd(@fun,@Jfun,x0,tol,nmax,verb);
for i = 1:length(rn)
    norm_x_Newton(1,i) = norm(rn(:,i));
end


fprintf('\n Broyden Method')
J_init = Jfun(x0);
B0 = J_init;
Bmat='fwd';
[r,rn] = broyden_method_nd(@fun,B0,x0,tol,nmax,Bmat,verb);
for i = 1:length(rn)
    norm_x_broyden(1,i) = norm(rn(:,i));
end

fprintf('\n\n Lazy Newton Method')
[r,rn] = lazynewton_method_nd(@fun,J_init,x0,tol,nmax,verb);
for i = 1:length(rn)
    norm_x_lazy(1,i) = norm(rn(:,i));
end


semilogy(1:length(norm_x_Newton),norm_x_Newton,1:length(norm_x_broyden),norm_x_broyden,1:length(norm_x_lazy),norm_x_lazy)
%semilogy(1:length(norm_x_broyden),norm_x_broyden)
%semilogy(1:length(norm_x_lazy),norm_x_lazy)
legend('Newton Method','Broyden Method','Lazy Newton Method')



function y = fun(x)

    y(1,1)=x(1)^2+x(2)^2-4;
    y(2,1)=exp(x(1))+x(2)-1;
end

function J = Jfun(x)
    J(1,1) = 2*x(1);
    J(1,2) = 2*x(2);
    J(2,1) = exp(x(1));
    J(2,2) = 1;
end