function [a,b] = scanoncorrFromInit(X,Y,cx,cy,aInit,bInit,param)

% [a,b] = scanoncorrFromInit(X,Y,cx,cy,aInit,bInit,param)
% Perform the alternating projected gradient algorithm for sparse 
% canonical correlation starting from an initial point. Adapted from
% Uurtio et al. (2019)
%   Uurtio, Viivi, Sahely Bhadra, and Juho Rousu. "Large-scale sparse 
%       kernel canonical correlation analysis." International Conference on
%       Machine Learning. PMLR, 2019.

maxIter = param.maxIter;
eps = param.eps;

a = aInit/norm(aInit,2);
b = bInit/norm(bInit,2);
a = projectL1(a,cx);
b = projectL1(b,cy);
Xa = X*a;
Yb = Y*b;

obj = ccaObjective(Xa,Yb);
objOld = obj;
iter = 0;
improvement = 42;

while improvement>eps && iter<maxIter

    % line search for a
    gradientA = ccaGrad(X,Xa,Yb);
    contA = 1;
    %gamma = norm(gradientA,2);
    gamma = norm(a,2);
    while contA && gamma>1e-10
        aNew = projectL1(a+gamma*gradientA,cx);
        XaNew = X*aNew;
        objNew = ccaObjective(XaNew,Yb);
        if objNew > obj + 1e-4*abs(obj)
            contA = 0;
            a = aNew;
            Xa = XaNew;
            obj = objNew;
        else
            gamma = gamma/2;
        end
    end
    % line search for b
    gradientB = ccaGrad(Y,Yb,Xa);
    contB = 1;
    %gamma = norm(gradientB,2);
    gamma = norm(b,2);
    while contB && gamma>1e-10
        bNew = projectL1(b+gamma*gradientB,cy);
        YbNew = Y*bNew;
        objNew = ccaObjective(Xa,YbNew);
        if objNew > obj + 1e-4*abs(obj)
            contB = 0;
            b = bNew;
            Yb = YbNew;
            obj = objNew;
        else
            gamma = gamma/2;
        end
    end

    improvement = (obj-objOld)/abs(obj+objOld);
    objOld = obj;
    iter = iter + 1;
end

%a = a/max(abs(a));
%b = b/max(abs(b));

if iter==maxIter
    warning('scanoncorrFromInit reached maximum number of iterations')
end