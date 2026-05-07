function output = cssd(x,y,p,gamma,xx,delta,varargin)
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
% output.pp: ppform of a smoothing spline with discontinuities. The pp's
%     domain extends one unit beyond [x_1, x_N] via linear extension, so
%     evaluations outside the data range are well-defined.
% output.yy: if xx is specified, holds ppval(output.pp, xx); otherwise [].
% output.discont: locations of detected discontinuities, the locations are a
%     subset of the midpoints of the data sites x.
% output.discont_idx: data-site index immediately before each discontinuity.
% output.interval_cell: a list of discrete indices between two discontinuities.
% output.pp_cell: a list of the cubic splines corresponding to the indices in
%     interval_cell. Each is linext-extended over its segment's domain.
% output.x, output.y: canonicalised data — duplicate sites have been
%     aggregated by weighted average, NaN/Inf rows have been dropped.
%     These may differ from the inputs when those operations apply.
% output.complexity_counter: number of times an input data point was visited
%     by the algorithm (proxy for runtime).
% output.rcv_score: rolling-CV score (sum of squared one-step-ahead linear
%     extrapolation residuals divided by max(N-2, 1)). FPVI extrapolates
%     from x_{blb}; PELT extrapolates from x_{rb}; both are scale-invariant
%     summaries useful for diagnostics.
% output.pcw_fun: PcwFunReal handle for convenient evaluation/plotting.
%
%   See also CSAPS, CSSD_CV

%%% BEGIN CHECK ARGUMENTS
if nargin<5, xx = []; end
if nargin<6, delta = []; end

if isempty(delta), delta = ones(size(x)); end

assert( (0 <= p) && (p <= 1), 'The p parameter must fulfill 0 <= p <= 1')
assert( 0 <= gamma, 'The gamma parameter must fulfill 0 <= gamma')   % N5: message matches the (non-strict) check

[xi, yi, wi, deltai] = chkxydelta(x, y, delta);

% Note: from now on we use the xi, yi, wi, deltai versions
%%% END CHECK ARGUMENTS

[N,D] = size(yi);

% rolling cv score
rcv_score = 0;

% counts the number of times an input data point is visited
% (for determination of computational complexity)
complexity_counter = 0;

% B6: validate the pruning string up front so misuse fails with a clear
% message rather than silently falling through to FPVI in the switch below.
parser = inputParser;
addOptional(parser, 'pruning', 'FPVI', ...
    @(s) ischar(s) && any(strcmp(s, {'FPVI', 'PELT'})));
parse(parser, varargin{:});
pruning = parser.Results.pruning;

% if gamma == Inf (discontinuity has infinite penalty), we may directly
% compute a classical smoothing spline
% also, if p == 1, we may straight compute an interpolating spline, no
% matter how large gamma is (smoothness costs are equal to 0)
if (gamma == Inf) || (p == 1)
    % N8: linext + embed_to_cubic so output.pp uses the same convention as
    % the DP branch (output.pp.breaks extends one unit beyond [x_1, x_N]).
    pp = csaps(xi,yi',p,[],wi);
    pp = linext_pp(pp, xi(1) - 1, xi(end) + 1);
    pp = embed_pptocubic(pp);
    discont = [];
    interval_cell = {1:N};
    pp_cell = {pp};
    complexity_counter = N;
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
        complexity_counter = N;

        %%% BEGIN MAIN LOOP
        for rb=2:N

            % best left bound (blb) initialized with 1 corresponding to interval 1:rb
            % corresponding Bellman value has been set in the precomputation
            blb = 1;

            A = [B(1:rb,:), rhs(1:rb,:)];

            eps_lr = 0;
            % the loop is performed in reverse order so that we may use pruning
            for lb = rb-1:-1:2
                complexity_counter = complexity_counter + 1;
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
        complexity_counter = N;
        %%% END PRECOMPUTATIONS
        
        % if gamma is hihger than the energy of the zero jump solution, we
        % dont need to go into the main loop
        if gamma >= F(end)
            pruning = 'FULL_SKIP';
        end

        %%% BEGIN MAIN LOOP

        switch pruning % two different pruning strategies are supported
            case 'FULL_SKIP'
                partition( end ) = 0;

            %%% BEGIN PELT Pruning
            case 'PELT'
                active_list = java.util.LinkedList();

                state_cell = cell(N, 3);
                for rb=2:N-1
                    % generates the initial states
                    [eps_lr, R, z] = startEpsLR(yi(rb:rb+1,:), d(rb), alpha(rb:rb+1), beta);
                    state_cell{rb, 1} = eps_lr;
                    state_cell{rb, 2} = R;
                    state_cell{rb, 3} = z;
                end
                active_list.add(2);
                for rb=3:N
                    % best left bound (blb) initialized with 1 corresponding to interval 1:rb
                    % corresponding Bellman value has been set in the precomputation
                    blb = 1;

                    % B1: remember which lb was the smallest active one BEFORE
                    % the iteration. Its state will be updated to the segment
                    % [first_lb, rb] inside the inner loop and used as the
                    % rolling-CV fallback if no candidate improves F(rb).
                    first_lb = active_list.peek();

                    listIterator = active_list.listIterator(active_list.size());
                    while (listIterator.hasPrevious())
                        lb = listIterator.previous();
                        eps_lr = state_cell{lb, 1};
                        R = state_cell{lb, 2};
                        z = state_cell{lb, 3};
                        if rb - lb > 1
                            [eps_lr, R, z] = updateEpsLR(eps_lr, R, yi(rb,:), d(rb-1), z, alpha(rb), beta);
                            state_cell{lb, 1} = eps_lr;
                            state_cell{lb, 2} = R;
                            state_cell{lb, 3} = z;
                            complexity_counter = complexity_counter + 1;
                        end
                        % check if setting a discontinuity between lb-1 and lb gives a
                        % better energy
                        candidate_value = F(lb-1) + gamma  + eps_lr;
                        if candidate_value < F(rb)
                            F(rb ) = candidate_value;
                            blb = lb;
                            stored_R = R;
                            stored_z = z;
                        end
                        %lb = active_arrlist(lb);
                    end

                    % B1: if no candidate improved F(rb) (blb still 1), fall back
                    % to first_lb's NOW-UPDATED state, which corresponds to the
                    % segment [first_lb, rb] — a valid segment ending at rb.
                    % (Previously stored_R/stored_z held leftover state from
                    % the precomputation loop, corresponding to [N-1, N].)
                    if blb == 1
                        stored_R = state_cell{first_lb, 2};
                        stored_z = state_cell{first_lb, 3};
                    end

                    % store the best left bound corresponding to the right bound rb
                    partition( rb ) = blb-1;

                    active_list.add(rb);

                    % PELT pruning
                    listIterator = active_list.listIterator(active_list.size());

                    while (listIterator.hasPrevious())
                        lb = listIterator.previous();
                        if F(lb-1) + state_cell{lb, 1} > F(rb)
                            listIterator.remove();
                        end
                    end
                    
                    

                    if rb < N
                        % Rolling-CV step: a_end, b_end are [f_{rb}, f'_{rb}]
                        % because PELT's QR feed absorbs the most recently added
                        % knot (rb) on the right, so the last 2 unknowns of the
                        % stored 4-unknown system are the right-edge values.
                        % We extrapolate linearly from x_{rb} to x_{rb+1}.
                        aux_ps = stored_R\stored_z;
                        a_end = aux_ps(end-1, :);
                        b_end = aux_ps(end, :);
                        rcv_score = rcv_score + sum( (a_end + b_end * (xi(rb+1) - xi(rb)) - yi(rb+1,:)).^2 );
                    end
                end

                % print for debugging
                %fprintf(['PELT pruned fraction:' num2str(1 - active_list.size()/N) '\n']);


                %%% END PELT PRUNING

            %%% BEGIN FPVI PRUNING
            case 'FPVI'
                for rb=3:N

                    % best left bound (blb) initialized with 1 corresponding to interval 1:rb
                    % corresponding Bellman value has been set in the precomputation
                    blb = 1;

                    % the loop is performed in reverse order so that we may use FPVVI-pruning
                    for lb = rb-1:-1:2

                        if lb == rb-1
                            complexity_counter = complexity_counter + 2;
                            % get start configuration and store start state in R, z
                            [eps_lr, R, z] = startEpsLR(yi([rb,rb-1],:), d(rb-1), alpha([rb,rb-1]), beta);
                            stored_R = R;
                            stored_z = z;
                        else
                            complexity_counter = complexity_counter + 1;
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
                            stored_R = R;
                            stored_z = z;
                        end

                    end
                    % store the best left bound corresponding to the right bound rb
                    partition( rb ) = blb-1;

                    % print for debugging
                    %fprintf(['rb:' num2str(rb) ', rb -lb:' num2str(rb - lb) '\n'])

                    if rb < N
                        % Rolling-CV step: in FPVI the QR feed is *reversed*
                        % (yi([rb,rb-1],:) at start, yi(lb,:) added each
                        % iteration), so the last 2 unknowns of the stored
                        % 4-unknown system are the LEFT edge of the segment
                        % whose state is in stored_R/stored_z. That segment
                        % is [blb, rb] when an improvement was found
                        % (blb >= 2), and the smallest 2-point segment
                        % [rb-1, rb] when no candidate improved F(rb)
                        % (blb stayed at 1 — stored_R was set at the very
                        % first inner-loop iteration via startEpsLR).
                        % Extrapolate from that left edge to x_{rb+1}.
                        % (PELT extrapolates from x_{rb}; both definitions
                        % are reasonable rolling-CV summaries.)
                        if blb >= 2
                            x_left = xi(blb);
                        else
                            x_left = xi(rb-1);
                        end
                        aux_ps = stored_R\stored_z;
                        a_end = aux_ps(end-1, :);
                        b_end = aux_ps(end, :);
                        rcv_score = rcv_score + sum( (a_end + b_end * (xi(rb+1) - x_left) - yi(rb+1,:)).^2 );
                    end
                end

                %%% END FPVI PRUNING

            otherwise
                error('cssd:UnknownPruning', ...
                    'Unknown pruning ''%s''. Expected ''FPVI'' or ''PELT''.', pruning);
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
            ymtx(end,:) = yi(interval, :);
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

% B3: avoid 0/0 NaN for N <= 2 (e.g. install_cssd's smoke test). The rolling-CV
% sum has N-2 contributions (from rb=3..N-1 in the inner loop, see below);
% for N <= 2 there are no contributions and the score is 0.
if N >= 3
    output.rcv_score = rcv_score / (N - 2);
else
    output.rcv_score = 0;
end

if isempty(xx)
    output.yy = [];
else
    output.yy = ppval(pp, xx);
end
output.xx = xx;
output.x = xi;
output.y = yi;

fun_cell = cell(numel(pp_cell),1);
for i=1:numel(pp_cell)
    fun_cell{i} = @(xx) ppval(pp_cell{i}, xx);
end
output.pcw_fun = PcwFunReal([-Inf; discont(:); Inf], fun_cell);
output.complexity_counter = complexity_counter;
%%% END SET OUTPUT

end


