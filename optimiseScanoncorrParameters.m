function [optInit,optCx,optCy,results] = ...
    optimiseScanoncorrParameters(X,Y,varargin)

%   [optInit,optCx,optCy,results] = optimiseScanoncorrParameters(X,Y)
%   Optimise scanoncorr hyperparameters using grid search and
%   cross-validation. The function will try different values of the
%   regularisation parameters and different initialisations, and
%   choose the best combination based on average test set correlation.
%
%   INPUTS:
%   X           -   n-by-px data matrix
%   Y           -   n-by-py data matrix
%   OPTIONAL INPUTS:
%   'gridSize'  -   size of the grid for cx and cy (default: 10)
%   'cxRange'   -   [min-cx-value, max-cx-value]
%                   (default: [1e-4*sqrt(px), sqrt(px)])
%   'cyRange'   -   [min-cy-value, max-cy-value]
%                   (default: [1e-4*sqrt(py), sqrt(py)])
%   'show'      -   Boolean, whether to display plots of results
%                   (default: 1)
%   'k'         -   cross-validation fold (default: 5)
%   'rounds'    -   how many rounds of cross-validation to perform
%                   (default: 5)
%   'D'         -   how many canonical vectors are found (default: 1)
%                   Note that the results will be averaged over all
%                   vectors
%   'init'      -   initialisation approach to use (by default both
%                   options are tried):
%                    'svd'      -   left and right singular vectors of the
%                                   cross-covariance matrix (default)
%                    'random'   -   random vector (see also 'rStarts')
%
%   OUTPUTS:
%   optInit     -   optimal initialisation method
%   optCx       -   optimal cx value
%   optCy       -   optimal cy value
%   results     -   structure with full results
%
%   EXAMPLE:
%      load carbig;
%      data = [Displacement Horsepower Weight Acceleration MPG];
%      nans = sum(isnan(data),2) > 0;
%      X = data(~nans,1:3); Y = data(~nans,4:5);
%      [optInit,optCx,optCy,results] = optimiseScanoncorrParameters(X,Y);

%   Author: T.Pusa, 2022

gridSize = 10;
cxRange = [1e-3*sqrt(size(X,2)) sqrt(size(X,2))];
cyRange = [1e-3*sqrt(size(Y,2)) sqrt(size(Y,2))];
show = 1;
k = 5;
rounds = 5;
D = 1;
init = 'all';

if ~isempty(varargin)
    if rem(size(varargin, 2), 2) ~= 0
		error('Check optional inputs.');
    else
        for i = 1:2:size(varargin, 2)
            switch varargin{1, i}
                case 'gridSize'
					gridSize = varargin{1, i+1};
                case 'cxRange'
					cxRange = varargin{1, i+1};
                case 'cyRange'
					cyRange = varargin{1, i+1};
                case 'show'
					show = varargin{1, i+1};
                case 'k'
					k = varargin{1, i+1};
                case 'rounds'
					rounds = varargin{1, i+1};
                case 'D'
					D = varargin{1, i+1};
                case 'init'
					init = varargin{1, i+1};
                    if ~ismember(init,["all";"svd";"random"])
                        error('No such initialisation option')
                    end
                otherwise
					error(['Could not recognise optional input names.' ...
                        '\nNo input named "%s"'],...
						varargin{1,i});
            end
        end
    end
end

xGrid = logspace(log10(cxRange(1)),log10(cxRange(2)),gridSize);
yGrid = logspace(log10(cyRange(1)),log10(cyRange(2)),gridSize);
results.xGrid = xGrid; results.yGrid = yGrid;
results.opt = 0;

%% random
if ismember(init,["all";"random"])
    score = zeros(gridSize);
    numA = zeros(gridSize);
    numB = zeros(gridSize);
    if show
        fprintf('Random init...\n');
    end
    parfor i=1:gridSize
        for j=1:gridSize
            if mod((i-1)*gridSize+j,10)==0 && show
                fprintf('\tCV %d of %d...\n', (i-1)*gridSize+j, gridSize^2);
            end
            [corrs,aTmp,bTmp] = crossValidateScanoncorr(X,Y,xGrid(i),yGrid(j),...
                'k',k,...
                'rounds',rounds,...
                'D',D,...
                'init','random',...
                'rStarts',5);
            score(i,j) = mean(abs(corrs),"all");
            numA(i,j) = mean(aTmp,"all");
            numB(i,j) = mean(bTmp,"all");
        end
    end
    random.score = score;
    [~, Ind] = max(score,[],"all");
    [I,J] = ind2sub(size(score),Ind);
    random.opt = score(I,J);
    random.optCx = xGrid(I);
    random.optCy = yGrid(J);
    random.numA = numA;
    random.numB = numB;
    results.random = random;
    if random.opt>results.opt
        results.opt = random.opt;
        results.optInit = 'random';
        results.optCx = random.optCx;
        results.optCy = random.optCy;
    end
    if show
        figure
        subplot(1,3,1)
        heatmap(yGrid,xGrid,score,"ColorLimits",[0 1],Colormap=jet)
        title('Correlation'); xlabel('cy'); ylabel('cx')
        subplot(1,3,2)
        heatmap(yGrid,xGrid,numA,"ColorLimits",[0 size(X,2)])
        title('A cardinality'); xlabel('cy'); ylabel('cx')
        subplot(1,3,3)
        heatmap(yGrid,xGrid,numB,"ColorLimits",[0 size(Y,2)])
        title('B cardinality'); xlabel('cy'); ylabel('cx')
        sgtitle('5x random init.')
        set(gcf,'Position',[100 100 1700 500])
    end
end

%% SVD
if ismember(init,["all";"svd"])
    score = zeros(gridSize);
    numA = zeros(gridSize);
    numB = zeros(gridSize);
    if show
        fprintf('SVD init...\n');
    end
    parfor i=1:gridSize
        for j=1:gridSize
            if mod((i-1)*gridSize+j,10)==0 && show
                fprintf('\tCV %d of %d...\n', (i-1)*gridSize+j, gridSize^2);
            end
            [corrs,aTmp,bTmp] = crossValidateScanoncorr(X,Y,xGrid(i),yGrid(j),...
                'k',k,...
                'rounds',rounds,...
                'D',D,...
                'init','svd');
            score(i,j) = mean(abs(corrs),"all");
            numA(i,j) = mean(aTmp,"all");
            numB(i,j) = mean(bTmp,"all");
        end
    end
    svd.score = score;
    [~, Ind] = max(score,[],"all");
    [I,J] = ind2sub(size(score),Ind);
    svd.opt = score(I,J);
    svd.optCx = xGrid(I);
    svd.optCy = yGrid(J);
    svd.numA = numA;
    svd.numB = numB;
    results.svd = svd;
    if svd.opt>results.opt
        results.opt = svd.opt;
        results.optInit = 'svd';
        results.optCx = svd.optCx;
        results.optCy = svd.optCy;
    end
    if show
        figure
        subplot(1,3,1)
        heatmap(yGrid,xGrid,score,"ColorLimits",[0 1],Colormap=jet)
        title('Correlation'); xlabel('cy'); ylabel('cx')
        subplot(1,3,2)
        heatmap(yGrid,xGrid,numA,"ColorLimits",[0 size(X,2)])
        title('A cardinality'); xlabel('cy'); ylabel('cx')
        subplot(1,3,3)
        heatmap(yGrid,xGrid,numB,"ColorLimits",[0 size(Y,2)])
        title('B cardinality'); xlabel('cy'); ylabel('cx')
        sgtitle('SVD init.')
        set(gcf,'Position',[110 110 1710 510])
    end
end

%%
optInit = results.optInit;
optCx = results.optCx;
optCy = results.optCy;
