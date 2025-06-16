classdef TestCSSD < matlab.unittest.TestCase

    properties
        Signals   % struct with fields x, y, delta (cell arrays)
    end

    properties (TestParameter)
        SigIdx = num2cell(1:7);    % adjust when you add/remove signals
        Gammas = num2cell(10.^(-10:4));
        Ps = num2cell(linspace(0, 1, 30));
    end

    methods (TestClassSetup)
        function genSignals(tc)
            
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
            g = @(x) besselj(1, 20 * x) + x .* ((0.3) <= x) .* (x <= 0.4) - x .* ((0.6) <= x) .* (x <= 1);
            N = 100;
            sigma = 0.1;
            delta = sigma * ones(N, 1);
            x = sort(rand(N,1));

            tc.Signals.x{end+1} = x; 
            tc.Signals.y{end+1} = g(x);
            tc.Signals.delta{end+1} = delta;
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
end
