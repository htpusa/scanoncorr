function [A,B,r,U,V] = scanoncorr(X,Y,cx,cy,varargin)

% SCANONCORR Sparse canonical correlation analysis
%   [A,B,r,U,V] = scanoncorr(X,Y,cx,cy) tries to find sparse canonical
%   coefficients for the data matrices X and Y using an alternating
%   projected gradient algorithm.
%   cx and cy are regularisation parameters for A and B respectively, such
%   that the L1-norm of each coefficient vector is equal to cx for A and cy
%   for B. A smaller value of cx, cy results in more sparsity.
%   The algorithm is largely adopted from 'gradKCCA' presented in Uurtio
%   et al. (2019).
%   Multiple coefficient vectors are found by deflating the data matrices
%   by projecting X,Y onto the orthocomplement of the space spanned by A,B
%   (see Mackey (2008)).
%
%   INPUTS:
%   X           -   n-by-px data matrix
%   Y           -   n-by-py data matrix
%   cx          -   regularisation parameter for A
%   cy          -   regularisation parameter for B
%   OPTIONAL INPUTS:
%   'D'         -   how many canonical vectors are found (default: 1)
%   'init'      -   how to initialise A and B:
%                    'svd'      -   left and right singular vectors of the
%                                   cross-covariance matrix (default)
%                    'random'   -   random vector (see also 'rStarts')
%   'rStarts'   -   how many random initialisations to perform 
%                   (default: 0 if 'init' is 'svd', 5 if 
%                   'init' is 'random')
%                   The function will seed A and B randomly 'rStarts' times
%                   and pick the highest objective value.
%   'maxIter'   -   maximum number of iterations
%   'eps'       -   tolerance parameter
%
%   OUTPUTS:
%   A           -   px-by-D matrix with canonical coefficients for X in
%                   columns
%   B           -   py-by-D matrix with canonical coefficients for Y in
%                   columns
%   r           -   1-by-D vector with the sample canonical correlations
%   U           -   n-by-D matrix with canonical variables/scores for X in
%                   columns
%   V           -   n-by-D matrix with canonical variables/scores for Y in
%                   columns
%
%   EXAMPLE:
%      load carbig;
%      data = [Displacement Horsepower Weight Acceleration MPG];
%      nans = sum(isnan(data),2) > 0;
%      X = data(~nans,1:3); Y = data(~nans,4:5);
%      [A,B] = scanoncorr(X,Y,1,1);

%   References:
%     Uurtio, Viivi, Sahely Bhadra, and Juho Rousu. "Large-scale sparse 
%       kernel canonical correlation analysis." International Conference on
%       Machine Learning. PMLR, 2019.
%     Mackey, Lester. "Deflation methods for sparse PCA." Advances in 
%       neural information processing systems 21 (2008).

%   Author: T.Pusa, 2022

D = 1;
init = 'svd';
rStarts = 0;
param.maxIter = 500;
param.eps = 1e-10;

if ~isempty(varargin)
    if rem(size(varargin, 2), 2) ~= 0
		error('Check optional inputs.');
    else
        for i = 1:2:size(varargin, 2)
            switch varargin{1, i}
                case 'D'
					D = varargin{1, i+1};
                case 'init'
					init = varargin{1, i+1};
                    if ~ismember(init,["svd";"random"])
                        error('No such initialisation option')
                    end
                case 'rStarts'
					rStarts = varargin{1, i+1};
                case 'maxIter'
					param.maxIter = varargin{1, i+1};
                case 'eps'
					param.eps = varargin{1, i+1};
                otherwise
					error(['Could not recognise optional input names.' ...
                        '\nNo input named "%s"'],...
						varargin{1,i});
            end
        end
    end
end

if init=="random" && rStarts<1
    rStarts = 5;
end

A = zeros(size(X,2),D);
B = zeros(size(Y,2),D);
r = zeros(1,D);
U = zeros(size(X,1),D);
V = zeros(size(Y,1),D);

X = X - mean(X,1);
Y = Y - mean(Y,1);

for i=1:D
    bestSoFar = 0;
    if init=="svd"
        [aInit,~,bInit] = svd(X'*Y);
        aInit = aInit(:,1);
        bInit = bInit(:,1);
        [A(:,i), B(:,i)] = scanoncorrFromInit(X,Y,cx,cy,aInit,bInit,param);
        bestSoFar = ccaObjective(X*A(:,i),Y*B(:,i));
    end
    for start=1:rStarts
        aInit = randn(size(X,2),1);
        bInit = randn(size(Y,2),1);
        [aCand, bCand] = scanoncorrFromInit(X,Y,cx,cy,aInit,bInit,param);
        objCand = ccaObjective(X*aCand,Y*bCand);
        if objCand > bestSoFar
            bestSoFar = objCand;
            A(:,i) = aCand;
            B(:,i) = bCand;
        end
    end
    r(i) = bestSoFar;
    U(:,i) = X*A(:,i);
    V(:,i) = Y*B(:,i);
    % deflate data
    X = deflate(X,A(:,i));
    Y = deflate(Y,B(:,i));
end