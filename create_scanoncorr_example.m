% Create simulated data as described in Witten et. al (2009)
%
%   Witten, Daniela M., Robert Tibshirani, and Trevor Hastie. 
%     "A penalized matrix decomposition, with applications to sparse 
%     principal components and canonical correlation analysis." 
%     Biostatistics 10.3 (2009): 515-534.

clear

n = 50;

a1 = [ones(20,1); -ones(20,1); zeros(60,1)];
a2 = [-ones(10,1); ones(10,1); -ones(10,1); ones(10,1); zeros(60,1)];
b1 = [zeros(60,1); -ones(20,1); ones(20,1)];
b2 = [zeros(60,1); ones(10,1); -ones(10,1); ones(10,1); -ones(10,1);];

Z = randn(n,2); W = orth(Z);
w1 = Z(:,1); w2 = Z(:,2);

X = normrnd(w1*a1' + w2*a2', 0.09); X = normalize(X);
Y = normrnd(w1*b1' + w2*b2', 0.09); Y = normalize(Y);

data.X = X;
data.Y = Y;
data.A = [a1 a2];
data.B = [b1 b2];

save('scanoncorr_example','data');



