function folds_cell = kfoldcv_split(N, K, random_state)
%KFOLDCV_SPLIT Returns a cell array of K-folds of signal of length N
% (each fold is stored as indices between 1 and N in ascending order)

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

