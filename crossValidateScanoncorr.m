function [corrs,numA,numB] = crossValidateScanoncorr(X,Y,cx,cy,varargin)

%   [corrs numA numB] = crossValidateScanoncorr(X,Y,cx,cy)
%   Perform cross-validation using the scanoncorr function. X and Y are
%   split into train and test sets repeatedly, canonical coefficients are 
%   found using the train set, and the correlation between projections is 
%   calculcated for the test set.
%
%   INPUTS:
%   X           -   n-by-px data matrix
%   Y           -   n-by-py data matrix
%   cx          -   regularisation parameter for A
%   cy          -   regularisation parameter for B
%   OPTIONAL INPUTS:
%   'k'         -   cross-validation fold (default: 2)
%   'rounds'    -   how many rounds of cross-validation to perform
%                   (default: 10)
%   'D'         -   how many canonical vectors are found (default: 1)
%   'init'      -   how to initialise A and B:
%                    'svd'      -   left and right singular vectors of the
%                                   cross-covariance matrix (default)
%                    'random'   -   random vector (see also 'rStarts')
%   'rStarts'   -   how many random initialisations to perform (default: 0)
%                   The function will seed A and B randomly 'rStarts' times
%                   and pick the highest objective value.
%
%   OUTPUTS:
%   corrs       -   'rounds'*'k'-by-'D' matrix of correlation coefficients
%   numA        -   'rounds'*'k'-by-'D' matrix of approximate cardinalities
%                   of A
%   numB        -   'rounds'*'k'-by-'D' matrix of approximate cardinalities
%                   of B
%
%   EXAMPLE:
%      load carbig;
%      data = [Displacement Horsepower Weight Acceleration MPG];
%      nans = sum(isnan(data),2) > 0;
%      X = data(~nans,1:3); Y = data(~nans,4:5);
%      corrs = crossValidateScanoncorr(X,Y,1,1);

%   Author: T.Pusa, 2022

k = 2;
rounds = 10;
D = 1;
init = 'svd';
rStarts = 0;

if size(X,1)~=size(Y,1)
    error('X and Y have a different number of samples')
end

if ~isempty(varargin)
    if rem(size(varargin, 2), 2) ~= 0
		error('Check optional inputs.');
    else
        for i = 1:2:size(varargin, 2)
            switch varargin{1, i}
                case 'k'
					k = varargin{1, i+1};
                case 'rounds'
					rounds = varargin{1, i+1};
                case 'D'
					D = varargin{1, i+1};
                case 'init'
					init = varargin{1, i+1};
                    if ~ismember(init,["svd";"random"])
                        error('No such initialisation option')
                    end
                case 'rStarts'
					rStarts = varargin{1, i+1};
                otherwise
					error(['Could not recognise optional input names.' ...
                        '\nNo input named "%s"'],...
						varargin{1,i});
            end
        end
    end
end

corrs = zeros(rounds*k,D);
numA = zeros(rounds*k,D);
numB = zeros(rounds*k,D);
for r=1:rounds
    part = crossvalind('KFold',size(X,1),k);
    for Ik=1:k
        train = part~=Ik;
        test = part==Ik;
        [A,B] = scanoncorr(X(train,:),Y(train,:),cx,cy,...
            'D',D,...
            'init',init,...
            'rStarts',rStarts);
        for d=1:D
            corrs((r-1)*k+Ik,d) = corr(X(test,:)*A(:,d), ...
                Y(test,:)*B(:,d));
            numA((r-1)*k+Ik,d) = sum(abs(A(:,d))>1e-2*max(abs(A(:,d))));
            numB((r-1)*k+Ik,d) = sum(abs(B(:,d))>1e-2*max(abs(B(:,d))));
        end
    end
end