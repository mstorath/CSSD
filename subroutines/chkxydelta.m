function [xi, yi, wi, deltai] = chkxydelta(x, y, delta)
% this is an auxiliary function for checking the input arguments of cssd
%
% Note (N1, audit): cssd's reconstruction loop also calls csaps with
% p == 0 for whichever segments arise from the DP partition. MATLAB's
% csaps internally returns the weighted-LS straight line in that case.
% This is a Curve Fitting Toolbox detail not visible from the cssd code.
%
% Note: this function depends on `chckxywp` (private function shipped
% with the Curve Fitting Toolbox). Without that toolbox, an "undefined
% function" error is raised below.

if isvector(y)
    y = y(:)';
end

[~, N] = size(y);
if N < 2
    error('There must be at least two data sites.')
end


if isempty(x)
    x = (1:N)';
end

if N ~= numel(x)
    error('x and y must have the same number of columns')
end

% Matlab uses the parameter w which is related to delta of De Boor's book by w = 1./delta.^2
if isempty(delta)
    delta = ones(N, 1);
end
w = delta.^(-2);


% remove Inf and NaN values
if isvector(y)
    valid_mask = isfinite(y);
else
    valid_mask = all(isfinite(y), 1);
end
y = y(:, valid_mask);
x = x(valid_mask);
w = w(valid_mask);

% checks arguments and creates column vectors (chckxywp is Matlab built in)
[xi,yi,~,wi] = chckxywp(x,y,2,w,0.5);
wi = wi(:);
deltai = sqrt(1./wi);
end