function output = cssd(x,y,p,gamma,xx,delta)
%CSSD Cubic smoothing spline with discontinuities
%
%   cssd(x, y, p, gamma, xx, delta) computes a cubic smoothing spline with discontinuities for the
%   given data (x,y). The data values may be scalars or vectors. Data points with the
%   same site are replaced by their (weighted) average as in the builtin csaps
%   function. 
%
% Input
% x: vector of data sites
%
% y: vector of same lenght as x or matrix where y(:,i) is a data vector at site x(i)
%
% p: parameter between 0 and 1 that weights the rougness penalty
% (high values result in smoother curves). Use CSSD_CV for automatic
% selection.
%
% gamma: parameter between 0 and Infinity that weights the discontiuity
% penalty (high values result in less discontinuities, gamma = Inf returns
% a classical smoothing spline). Use CSSD_CV for automatic
% selection.
%
% xx: (optional) evaluation points for the result
%
% delta: (optional) weights of the data sites. delta may be thought of as the
% standard deviation of the at site x_i. Should have the same size as x.
% - Note: The Matlab built in spline function csaps uses a different weight
% convention (w). delta is related to Matlab's w by w = 1./delta.^2
% - Note for vector-valued data: Weights are assumed to be identical over
% vector-components. (Componentwise weights might be supported in a future version.)
%
% Output
% output = cssd(...)
% output.pp: ppform of a smoothing spline with discontinuities; if xx is specified,
% the evaluation of the result at the points xx is returned
% output.discont: locations of detected discontinuities, the locations are a
% subset of the midpoints of the data sites x
% output.interval_cell: a list of discrete indices between two discontinuities
% output.pp_cell: a list of the cubic splines corresponding to the indices in interval_cell
%
%   See also CSAPS, CSSD_CV

%%% BEGIN CHECK ARGUMENTS
if nargin<5, xx = []; end
if nargin<6, delta = []; end

if isempty(delta), delta = ones(size(x)); end

assert( (0 <= p) && (p <= 1), 'The p parameter must fulfill 0 <= p <= 1')
assert( 0 <= gamma, 'The gamma parameter must fulfill 0 < gamma')

% Matlab uses the parameter w which is related to delta of De Boor's book by w = 1./delta.^2
w = 1./delta.^2;

% checks arguments and creates column vectors (chckxywp is Matlab built in)
[xi,yi,~,wi] = chckxywp(x,y,2,w,p);
deltai = sqrt(1./wi);

% Note: from now on we use the xi, yi, wi, deltai versions
%%% END CHECK ARGUMENTS

[N,D] = size(yi);

% if gamma == Inf (discontinuity has infinite penalty), we may directly
% compute a classical smoothing spline
% also, if p == 1, we may straight compute an interpolating spline, no
% matter how large gamma is (smoothness costs are equal to 0)
if (gamma == Inf) || (p == 1)
    pp = csaps(xi,yi',p,[],wi);
    discont = [];
    interval_cell = {1:N};
    pp_cell = {fnxtr(pp,2)};
else
    % F stores Bellmann values
    F = zeros(N, 1);
    % partition: stores the optimal partition
    partition = zeros(N, 1);

    %%% BEGIN PIECEWISE LINEAR CASE
    if p == 0 % the piecewise linear case
        B = [ones(N,1), xi]./deltai(:);
        rhs = yi./deltai(:);
        % precompute eps_1r for r=1,...,N
        A = [B, rhs];
        G = planerot(A(1:2,1));
        A(1:2, :) = G*A(1:2, :);
        eps_1r = 0;
        % loop starts from index three because eps_11 and eps_12 are zero
        for r=3:N
            G = planerot(A([1,r],1));
            A([1,r],:) = G * A([1,r],:);
            G = planerot(A([2,r],2));
            A([2,r],2:end) = G * A([2,r],2:end);
            eps_1r = eps_1r + sum(A(r, 3:end).^2);
            % store the eps_1r as the initial Bellman value corresponding to a
            % solution without discontinuities
            F(r) = eps_1r;
        end
        %%% BEGIN MAIN LOOP
        for rb=2:N

            % best left bound (blb) initialized with 1 corresponding to interval 1:rb
            % corresponding Bellman value has been set in the precomputation
            blb = 1;

            A = [B(1:rb,:), rhs(1:rb,:)];

            eps_lr = 0;
            % the loop is performed in reverse order so that we may use pruning
            for lb = rb-1:-1:2
                if lb == rb-1
                    G = planerot(A([end,end-1],1));
                    A([end,end-1], :) = G*A([end,end-1], :);
                else
                    G = planerot(A([end, lb],1));
                    A([end,lb], :) = G * A([end,lb], :);
                    G = planerot(A([end-1, lb],2));
                    A([end-1,lb], 2:end) = G*A([end-1,lb], 2:end);
                    eps_lr = eps_lr + sum(A(lb,3:end).^2);
                end

                % check if setting a discontinuity between lb-1 and lb gives a
                % better energy
                candidate_value = F(lb-1) + gamma  + eps_lr;
                if candidate_value < F(rb)
                    F(rb ) = candidate_value;
                    blb = lb;
                end
                % store the best left bound corresponding to the right bound rb
                partition( rb ) = blb-1;
            end
        end
        %%% END MAIN LOOP

        %%% END PIECEWISE LINEAR CASE


        %%% BEGIN CSSD CASE
    else % this is the standard case: gamma > 0 and 0 < p < 1
        %%% BEGIN PRECOMPUTATIONS
        beta = sqrt(1-p);
        alpha = sqrt(p)./deltai;
        d = diff(xi); % xi is sorted ascendingly

        % precompute eps_1r for r=1,...,N
        [eps_1r, R, z] = startEpsLR(yi(1:2,:), d(1), alpha(1:2), beta);
        % loop starts from index three because eps_11 and eps_12 are zero
        for r=3:N
            [eps_1r, R, z] = updateEpsLR(eps_1r, R, yi(r,:), d(r-1), z, alpha(r), beta);
            % store the eps_1r as the initial Bellman value corresponding to a
            % solution without discontinuities
            F(r) = eps_1r;
        end
        %%% END PRECOMPUTATIONS

        %%% BEGIN MAIN LOOP
        for rb=2:N

            % best left bound (blb) initialized with 1 corresponding to interval 1:rb
            % corresponding Bellman value has been set in the precomputation
            blb = 1;

            % the loop is performed in reverse order so that we may use pruning
            for lb = rb-1:-1:2
                if lb == rb-1
                    % get start configuration and store start state in R, z
                    [eps_lr, R, z] = startEpsLR(yi([rb,rb-1],:), d(rb-1), alpha([rb,rb-1]), beta);
                else
                    % perform fast energy update and store current state in R, z
                    [eps_lr, R, z] = updateEpsLR(eps_lr, R, yi(lb,:), d(lb), z, alpha(lb), beta);
                end

                % pruning to skip unreachable configurations
                % (if this condition is met the following if-condition cannot never
                % be fulfilled because eps_lr is monote increasing and F >= 0.)
                if (eps_lr + gamma) >= F(rb)
                    break
                end

                % check if setting a discontinuity between lb-1 and lb gives a
                % better energy
                candidate_value = F(lb-1) + gamma  + eps_lr;
                if candidate_value < F(rb)
                    F(rb ) = candidate_value;
                    blb = lb;
                end

            end
            % store the best left bound corresponding to the right bound rb
            partition( rb ) = blb-1;
        end
        %%% END MAIN LOOP


    end
    %%% END CSSD CASE

    %%% BEGIN RECONSTRUCTION
    % the discontinuity locations are coded in the array 'partition'. The
    % vector [partition(rb)+1:rb] gives the indices of between two
    % discontinuity locations. We start from behind with [partition(N)+1:N] and
    % successively compute the preceding intervals.
    rb = N;
    pp_cell = {};
    interval_cell = {};
    discont = [];
    upper_discont = xi(end) + 1;
    while rb > 0
        % partition(rb) stores corresponding optimal left bound lb
        lb = partition(rb);
        if lb == 0
            lower_discont = xi(1) - 1;
        else
            lower_discont = (xi(lb+1) + xi(lb)) /2;
        end
        interval = (lb+1) : rb;
        interval_cell{end+1} = interval; %#ok<AGROW> (runtime not critical in this part of the algorithm)
        if length(interval) == 1 % this case should happen rarely but may happen e.g. for data of uneven length and low gamma parameter
            ymtx = zeros(4,D);
            ymtx(:,D) = yi(interval, :);
            pp = ppmak([lower_discont, upper_discont], ymtx', D);
        else
            pp = csaps(xi(interval),yi(interval, :)', p, [], wi(interval));
            pp = linext_pp(pp, lower_discont, upper_discont);
            pp = embed_pptocubic(pp);
        end
        pp_cell{end+1} = pp; %#ok<AGROW> (runtime not critical in this part of the algorithm)
        % continue with next right bound
        rb = lb;
        upper_discont = lower_discont;
        discont(end+1) = lower_discont; %#ok<AGROW> (runtime not critical in this part of the algorithm)
    end
    %%% END RECONSTRUCTION

    %%% BEGIN MAKE PP FORM
    pp_cell = flip(pp_cell); % the pp's were computed in reverse order which is fixed here
    interval_cell = flip(interval_cell);
    pp = merge_ppcell(pp_cell);
    %%% END MAKE PP FORM

    discont = flip(discont(1:end-1))'; % the discontinuities were computed in reverse order which is fixed here

end


%%% BEGIN SET OUTPUT
output.pp = pp;
output.discont = discont;
output.interval_cell = interval_cell;
output.pp_cell = pp_cell;
output.discont_idx = zeros(numel(interval_cell)-1, 1);
for i = 1:numel(output.discont_idx)
    output.discont_idx(i) = output.interval_cell{i}(end);
end

if isempty(xx)
    output.yy = [];
else
    output.yy = ppval(pp, xx);
end

fun_cell = cell(numel(pp_cell),1);
for i=1:numel(pp_cell)
    fun_cell{i} = @(xx) ppval(pp_cell{i}, xx);
end
output.pcw_fun = PcwFunReal([-Inf; discont(:); Inf], fun_cell);
%%% END SET OUTPUT

end

