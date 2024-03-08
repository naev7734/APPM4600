function lvec = lxfun(xint,xeval)
    n = length(xint);
    lvec = ones(n);

    for i = 1:n
        for j = 1:n
            if i~=j
                l(i) = l(i) * (xeval-x(j))/(x(i)-x(j));
            end
        end
    end


end