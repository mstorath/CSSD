classdef TestCSSD < matlab.unittest.TestCase

    properties
        Signals   % struct with fields x, y, delta (cell arrays)
    end

    properties (TestParameter)
        SigIdx = num2cell(1:8);    % adjust when you add/remove signals
        Gammas = num2cell(10.^(-10:4));
        Ps = num2cell(linspace(0, 1, 30));
    end

    methods (TestClassSetup)
        function genSignals(tc)
            rng(123)

            % short signals
            tc.Signals.y     = { ...
                [0 1 1],...
                [1 0 1],...
                [1 1 0],...
                [0,0,1,1],...
                [0,0,0,1,1,1], ...
                [0,0,1,1,2,2], ...
                [0,1,0,1,0,1], ...
                };
            k = length(tc.Signals.y);
            tc.Signals.x = cell(k,1);
            for i = 1:k
                n = length(tc.Signals.y{i});
                tc.Signals.x{i}     = 1:n;
                tc.Signals.delta{i} = ones(1, n);
            end
            
            % longer signals
            funcs{1} = @(x) besselj(1, 20 * x) + x .* ((0.3) <= x) .* (x <= 0.4) - x .* ((0.6) <= x) .* (x <= 1);
            funcs{2} = @(x) 4.*sin(4*pi.*x) - sign(x - .3) - sign(.72 - x);

            for i = 1:numel(funcs)
                g = funcs{i};
                N = 100;
                sigma = 0.1;
                delta = sigma * ones(N, 1);
                x = sort(rand(N,1));
    
                tc.Signals.x{end+1} = x; 
                tc.Signals.y{end+1} = g(x);
                tc.Signals.delta{end+1} = delta;
            end


        end
    end

    methods (Test)
        function prunings(tc, SigIdx, Gammas, Ps)
            p     = Ps;
            gamma = Gammas;

            x     = tc.Signals.x{    SigIdx};
            y     = tc.Signals.y{    SigIdx};
            delta = tc.Signals.delta{SigIdx};

            outFPVI = cssd(x, y, p, gamma, [], delta, 'pruning', 'FPVI');
            outPELT = cssd(x, y, p, gamma, [], delta, 'pruning', 'PELT');
            tc.verifyEqual( ...
                outFPVI.pp, outPELT.pp, ...
                'AbsTol', 1e-12, ...
                sprintf('Pruning mismatch in signal %d.', SigIdx) );

        end
    end

    %% Regression tests for the audit fixes (B1-B8 + N1-N8).
    % Each test name begins with `test_<id>_` so it can be cross-referenced
    % with the corresponding entry in PORTING_NOTES.md.
    methods (Test)
        function test_B1_pelt_rcv_score_finite_on_constant_signal(tc)
            % B1: with effectively-Inf gamma, no candidate improves F(rb)
            % in the PELT main loop, so stored_R/stored_z previously held
            % stale state from the precompute. After the fix, rcv_score
            % must be a finite real number.
            x = 1:5; y = [0 0 0 0 0]; delta = ones(1,5);
            out = cssd(x, y, 0.5, 1e6, [], delta, 'pruning', 'PELT');
            tc.verifyTrue(isfinite(out.rcv_score), ...
                'PELT rcv_score should be finite even when no improvement');
        end

        function test_B2_rolling_cv_finite_both_pruners(tc)
            % B2: both prunings should produce finite rcv_score on a
            % non-trivial signal, regardless of which extrapolation
            % convention they use.
            x = 1:10; y = [0 0 0 0 0 1 1 1 1 1]; delta = ones(1,10);
            for pr = {'FPVI', 'PELT'}
                out = cssd(x, y, 0.5, 0.1, [], delta, 'pruning', pr{1});
                tc.verifyTrue(isfinite(out.rcv_score), ...
                    sprintf('rcv_score not finite for %s', pr{1}));
            end
        end

        function test_B3_rcv_score_no_nan_for_n_eq_2(tc)
            % B3: the rcv_score / (N-2) divisor produced 0/0 = NaN for N=2.
            out = cssd([0,1], [0,0], 1, 1);
            tc.verifyFalse(isnan(out.rcv_score), ...
                'rcv_score must be 0 (not NaN) for N=2');
        end

        function test_B4_cssd_cv_does_not_crash_on_default_call(tc)
            % B4 / B5: cssd_cv with no optional arguments must run without
            % crashing or warning (the `p = inputParser` shadow used to
            % be a latent fragility rather than an error).
            rng(0);
            x = (1:20).';
            y = sin(0.5 * x) + 0.05 * randn(20, 1);
            cv = cssd_cv(x, y);
            tc.verifyTrue(isfield(cv, 'p') && isfinite(cv.p));
            tc.verifyTrue(isfield(cv, 'gamma') && isfinite(cv.gamma));
        end

        function test_B5_cssd_cv_random_default_K(tc)
            % B5: cssd_cv(x, y, 'random') without specifying K must use
            % the K=5 default rather than crashing.
            rng(0);
            x = (1:20).';
            y = sin(0.5 * x) + 0.05 * randn(20, 1);
            cv = cssd_cv(x, y, 'random');
            tc.verifyTrue(isfinite(cv.cv_score));
        end

        function test_B6_invalid_pruning_errors(tc)
            % B6: previously, an unknown pruning string silently used FPVI
            % (the implicit `otherwise` case). Now it must error at
            % parse time with a descriptive identifier.
            tc.verifyError( ...
                @() cssd([1 2 3], [0 1 0], 0.5, 1, [], [], 'pruning', 'pelt'), ...
                ?MException);
            tc.verifyError( ...
                @() cssd([1 2 3], [0 1 0], 0.5, 1, [], [], 'pruning', 'BOGUS'), ...
                ?MException);
        end

        function test_B7_pcw_fun_vector_valued_shape(tc)
            % B7: PcwFunReal.eval must return correct shape (numel(xx), dim)
            % for vector-valued data. The original `yy = NaN(size(xx))`
            % allocation made dim>1 either error or silently truncate.
            rng(0);
            x = (1:10).';
            y = [sin(x), cos(x)].';   % 2 x 10 (D x N as cssd expects)
            delta = ones(10, 1);
            out = cssd(x, y, 0.99, Inf, [], delta);
            xx = linspace(min(x), max(x), 50).';
            v = out.pcw_fun.eval(xx);
            tc.verifySize(v, [50, 2]);
            tc.verifyTrue(all(isfinite(v(:))), ...
                'pcw_fun.eval should produce finite values inside data range');
        end

        function test_B7_pcw_fun_scalar_unchanged(tc)
            % B7 regression: scalar case must continue to work and return
            % a (numel(xx), 1) array (even though previously it returned
            % size(xx)). Documented as the new convention.
            x = 1:6; y = [0 0 1 1 0 0];
            out = cssd(x, y, 0.99, 0.5);
            xx = linspace(1, 6, 30).';
            v = out.pcw_fun.eval(xx);
            tc.verifySize(v, [30, 1]);
        end

        function test_B8_cssd_cv_robust_at_boundary(tc)
            % B8: gamma_pq used to produce NaN at q=1 when p=0, which the
            % SA optimiser could occasionally sample. With the q upper
            % bound and the eps guard, cv_score must always be finite or
            % +Inf (never NaN).
            rng(0);
            x = (1:10).';
            y = randn(10, 1);
            % Seed an obviously-bad starting point to exercise the boundary.
            cv = cssd_cv(x, y, 'random', 5, [], [0; 1e10]);
            tc.verifyTrue(~isnan(cv.cv_score));
        end

        function test_N3_merge_endpoint_mismatch_errors(tc)
            % N3: merge_ppcell with mismatched endpoints must raise a clear
            % error identifier (was previously silent).
            pp1 = csaps([0 1 2], [0 1 0]);    % ends at 2
            pp2 = csaps([3 4 5], [0 1 0]);    % starts at 3 — mismatch
            tc.verifyError(@() merge_ppcell({pp1, pp2}), ...
                'merge_ppcell:EndpointMismatch');
        end

        function test_N6_kfoldcv_invalid_K_errors(tc)
            % N6: K must satisfy 2 <= K <= N. Previously K=1, K=0, K>N
            % silently produced degenerate folds.
            tc.verifyError(@() kfoldcv_split(5, 1),  'kfoldcv_split:InvalidK');
            tc.verifyError(@() kfoldcv_split(5, 6),  'kfoldcv_split:InvalidK');
            tc.verifyError(@() kfoldcv_split(5, 0),  'kfoldcv_split:InvalidK');
        end

        function test_N8_pp_shape_consistent_between_branches(tc)
            % N8: output.pp.breaks now uses the same convention in the
            % gamma=Inf branch as in the DP branch (both linext-extended
            % one unit beyond [x_1, x_N]).
            x = (1:5).'; y = (1:5).';
            out_inf = cssd(x, y, 0.99, Inf);
            out_dp  = cssd(x, y, 0.99, 1e10);   % DP path, but no jumps
            tc.verifyEqual(numel(out_inf.pp.breaks), numel(out_dp.pp.breaks), ...
                'output.pp.breaks length differs between Inf and DP branches');
            % Both should extend exactly one unit beyond the data range.
            tc.verifyEqual(out_inf.pp.breaks(1), x(1) - 1, 'AbsTol', 1e-12);
            tc.verifyEqual(out_inf.pp.breaks(end), x(end) + 1, 'AbsTol', 1e-12);
        end
    end
end
