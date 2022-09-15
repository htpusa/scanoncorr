function X = deflate(X,a)

% X = deflate(X,a)
% Project X onto the orthocomplement of the space spanned by a.

normA = a/norm(a,2);
X = X - (normA*normA'*X')';