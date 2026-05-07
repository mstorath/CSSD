function folds_cell = kfoldcv_split(N, K, random_state)
%KFOLDCV_SPLIT Returns a cell array of K-folds of signal of length N
% (each fold is stored as indices between 1 and N in ascending order)
%
% N6 (audit): K must satisfy 2 <= K <= N. K=1 leaves no training data
% (cssd would be called with the empty set); K>N produces some empty folds.

assert(K >= 2 && K <= N, ...
    'kfoldcv_split:InvalidK', ...
    'K must satisfy 2 <= K <= N (got K=%d, N=%d).', K, N);

if nargin < 3
    random_state = [];
end
if ~isempty(random_state)
    rng(random_state)
end

% create folds
ridx = randperm(N);
folds_cell = cell(K,1);
for k = 1:K
    folds_cell{k} = sort(ridx(k:K:end));
end

end

