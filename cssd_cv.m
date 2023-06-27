function output = cssd_cv(x, y, cv_type, cv_arg, delta, startingPoint, varargin)
%CSSD_CV K-Fold cross validation for cubic smoothing spline with discontinuities 
%
%   cssd_cv(x, y, cv_type, cv_arg, delta, startingPoint, varargin)  automatically determines
% the model parameters p and gamma of a CSSD model based on minimization of 
% a K-fold cross validation score. 
% 
% Note: The minimization process uses standard derivative free
% optimizers (simulated annealing and Nelder-Mead simplex downhill). So it
% is not guaranteed that the result is a global minimum. 
% Trying different starting points often helps if the results with the default starting point 
% are unsatisfactory.
%
% Input
% x: vector of data sites
%
% y: vector of same lenght as x or matrix where y(:,i) is a data vector at site x(i)
%
% cv_type (optional): 'random', 'equi', or 'custom'
% cv_arg (optional): a cell array of index vectors corresponding to the
% K folds (only relevant for CV-type 'custom')
% 
% startingPoint (optional): Starting point startingPoint = [p; gamma] for the optimizer
% (default is [0.5; 1])
%
% delta: (optional) weights of the data sites. delta may be thought of as the
% standard deviation of the at site x_i. Should have the same size as x.
% - Note: The Matlab built in spline function csaps uses a different weight
% convention (w). delta is related to Matlab's w by w = 1./delta.^2
% - Note for vector-valued data: Weights are assumed to be identical over
% vector-components. (Componentwise weights might be supported in a future version.)
%
% Output
% output = cssd_cv(...)
% output.p: % p: parameter between 0 and 1 that weights the rougness penalty
% (high values result in smoother curves)
% output.gamma: parameter between 0 and Infinity that weights the discontiuity
% penalty (high values result in less discontinuities, gamma = Inf corresponds to a classical smoothing spline
% output.cv_score: best cv_score achieved by the optimizer
% output.cv_fun: a function handle to the CV scoring function
%
%   See also CSSD


if nargin < 3
    cv_type = [];
end
if nargin < 4
    cv_arg = [];
end
if nargin < 5
    delta = [];
end
if nargin < 6
    startingPoint = [];
end

p = inputParser;
addOptional(p,'verbose', 0);
addOptional(p,'maxTime', Inf);
addOptional(p,'pruning', 'FPVI');
parse(p,varargin{:});
verbose = p.Results.verbose;
maxTime = p.Results.maxTime;
pruning_method = p.Results.pruning;

if isempty(delta)
    delta = ones(size(x));
end

%check data
[xi, yi, ~, deltai] = chkxydelta(x, y, delta);

[N,D] = size(yi);

% create folds if not given
if isempty(cv_type)
    cv_type = 'random';
    cv_arg = 5;
end

switch cv_type
    case 'random'
        folds_cell = kfoldcv_split(N, cv_arg);
    case 'equi'
        folds_cell = cell(cv_arg, 1);
        for i = 1:cv_arg
            folds_cell{i} = i:cv_arg:N;
        end
    case 'custom'
        folds_cell = cv_arg;
    otherwise
        error('This CV method is unknown.')
end

%gamma_tf = @(b) atan(b) * 2/pi;
%gamma_itf = @(a) (tan(a * pi/2));
% parametrize gamma = p * q/(1-q)
gamma_pq = @(p,q) p * (q / (1-q));

% generate cv score function
cv_fun = @(p, gamma) cssd_cvscore(xi, yi, p, gamma, deltai, folds_cell, pruning_method);
%cv_fun_vec = @(z) cv_fun(z(1), gamma_itf(z(2))); % vectorised and transformed version for optimization

cv_fun_vec = @(z) cv_fun(z(1), gamma_pq(z(1), z(2))); % vectorised and transformed version for optimization

% perform optimization
saoptions = {@simulannealbnd,'MaxTime', maxTime};
options = {};
if verbose
    saoptions = {saoptions{:},  'Display','iter','PlotFcns', {@saplotbestx,@saplotbestf,@saplotx,@saplotf}};
    options = {options{:}, 'Display','iter', 'PlotFcns', {@optimplotx, @optimplotfunccount, @optimplotfval}};
else
    saoptions = {saoptions{:},  'Display','off'};
    options = {options{:}, 'Display','off'};
end



% set starting point of optimizers
if isempty(startingPoint)
    %startingPoint = [0.5; 1]; % starting values: p =0.5, gamma = 1
    startingPoint_tf = [0.5; 0.5]; % starting values: p =0.5, q = 0.5
else
    p = startingPoint(1);
    gamma = startingPoint(2);
    startingPoint_tf = [p; gamma / (p + gamma)];
end

% transform starting point to lie in the unit square [0,1]^2
%startingPoint_tf = [startingPoint(1); gamma_tf(startingPoint(2))];
%startingPoint_tf = [startingPoint(1); startingPoint(1) * gamma_tf(startingPoint(2))];


% invoke chain of standard derivative-free optimizers on [0,1]^2: simulated
% annealing followed by Nelder-Mead simplex downhill
[improvedPoint_tf, cv_score] = simulannealbnd(cv_fun_vec, startingPoint_tf, [0;0], [1;1], optimoptions(saoptions{:})); % simulated annealing
[improvedPoint_tf, cv_score] = fminsearch(cv_fun_vec, improvedPoint_tf, optimset(options{:})); % refine result of simulated annealing by Nelder Mead downhill

% improved p and gamma
p = improvedPoint_tf(1);
%gamma = gamma_itf(improvedPoint_tf(2)); % transform gamma back to [0, Inf)
%gamma = gamma_itf(improvedPoint_tf(2)/improvedPoint_tf(1)); % transform gamma back to [0, Inf)
gamma = gamma_pq(improvedPoint_tf(1), improvedPoint_tf(2));

% set ouptput
output.p = p;
output.gamma = gamma;
output.cv_score = cv_score;
output.cv_fun = cv_fun;


end





