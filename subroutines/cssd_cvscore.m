function cv_score = cssd_cvscore(x, y, p, gamma, delta, folds_cell)
% cssd_cvscore Computes the K-fold cross validation score of a CSSD model
% with parameters p and gamma for data (x,y)
%
% Input: 
% x, y, p, gamma, delta: See CSSD function
% folds_cell: a cell array of K index vectors decribing the K folds of the data
%
% Output: the K-fold crossvalidation score of the specified CSSD model
%
% See also: CSSD, CSSD_CV

K = numel(folds_cell);
N = numel(x);
if (p < 0) || (p > 1) || (gamma <= 0) % parameters outside bounds
    cv_score = Inf;
    return
else
    cv_score = 0;
    for k=1:K
        test_idx = folds_cell{k}; % indices to leave out
        train_idx = setdiff(1:N, test_idx); % indices to estimate from
        output = cssd(x(train_idx), y(train_idx,:), p, gamma, [], delta(train_idx)); % estimate on training set
        pp = output.pp;
        cv_score = cv_score + sum( ((ppval(pp, x(test_idx)) - y(test_idx, :))./delta(train_idx)).^2, 'all' ); % compute cv score
    end
    cv_score = cv_score/N;
end
end