function cv_score = cssd_cvscore(x, y, p, gamma, delta, folds_cell, pruning_method)
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
        output = cssd(x(train_idx), y(train_idx,:)', p, gamma, [], delta(train_idx), 'pruning', pruning_method); % estimate on training set
        pp = output.pp;
        pp_y_test = ppval(pp, x(test_idx)')'; % transposes to assure that pp_y_test has second dim = N (accounts for a special behaviour of ppval)
        y_test = y(test_idx, :);
        delta_test = delta(test_idx);
        assert(size(pp_y_test,1) == size(y_test,1))
        assert(all(size(pp_y_test,1) == size(delta_test,1)))
        cv_score = cv_score + sum( ((pp_y_test - y_test)./delta_test).^2, 'all' ); % compute cv score
        
    end
    cv_score = cv_score/N;
end
end