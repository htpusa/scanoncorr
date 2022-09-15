function grad = ccaGrad(X,Xa,Yb)

% grad = ccaGrad(X,Xa,Yb)
% Gradient of CCA objective with respect to a

XaN = norm(Xa,2);
YbN = norm(Yb,2);

grad = (X'*Yb - (Xa'*Yb) * (X'*Xa)/XaN^2) / (XaN*YbN);