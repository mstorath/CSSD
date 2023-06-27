% Example: Runtime for different types of signals

% for comparison the Python module `ruptures` has to be installed via "pip install ruptures"
% change active folder to demos folder if module does not load
ruptures_cssd = py.importlib.import_module('ruptures_cssd');
py.importlib.reload(py.importlib.import_module('ruptures_cssd'));
np = py.importlib.import_module('numpy');

% HeaviSine signal
g = @(x) 4.*sin(4*pi.*x) - sign(x - .3) - sign(.72 - x);

% parameters
p = 0.9999;
gamma = 20;
sigma = 0.4;

% signal lengths
signal_lengths_mult = [1,2,4,8,16,32];
signal_lengths = signal_lengths_mult*250;

% number of runs
K = 3;

% space for runtimes 
runtimes_dense_proposed_fpvi = zeros(numel(signal_lengths), K);
runtimes_rep_proposed_fpvi = zeros(numel(signal_lengths), K);
runtimes_dense_proposed_pelt = zeros(numel(signal_lengths), K);
runtimes_rep_proposed_pelt = zeros(numel(signal_lengths), K);
runtimes_dense_baseline = zeros(numel(signal_lengths), K);
runtimes_rep_baseline = zeros(numel(signal_lengths), K);

% space for number of times input data is visited
count_dense_proposed_fpvi = zeros(numel(signal_lengths), K);
count_rep_proposed_fpvi = zeros(numel(signal_lengths), K);
count_dense_proposed_pelt = zeros(numel(signal_lengths), K);
count_rep_proposed_pelt = zeros(numel(signal_lengths), K);
count_dense_baseline = zeros(numel(signal_lengths), K);
count_rep_baseline = zeros(numel(signal_lengths), K);


for k = 1:K
    disp(['Run = ', num2str(k), ' of ', num2str(K)])
    % EXPERIMENT 1: DENSER SAMPLING, SAME NUMBER OF DISCONTINUITIES
    disp('  Ex 1')
    for i=1:length(signal_lengths)
        N = signal_lengths(i);
        disp(['    Processing N = ', num2str(N)]);
        x1 = (0:N-1)'/(N-1);
        y1 = g(x1) + sigma .* randn(N, 1);
        delta = sigma * ones(N,1);
        
        % Baseline
        output_baseline1 = ruptures_cssd.detect_changepoints(np.array(x1), np.array(y1), p, gamma, np.array(delta));
        runtimes_dense_baseline(i, k) = output_baseline1{2};
        count_dense_baseline(i, k) = output_baseline1{4};
        disp(['    RT Baseline: ', num2str(runtimes_dense_baseline(i, k)), ' sec']);
        
        % Proposed
        tic
        output_dense_cssd_fpvi = cssd(x1, y1, p, gamma, [], delta, 'pruning', 'FPVI');
        runtimes_dense_proposed_fpvi(i, k) = toc;
        count_dense_proposed_fpvi(i, k) = output_dense_cssd_fpvi.complexity_counter;
        disp(['    RT Proposed + FPVI: ', num2str(runtimes_dense_proposed_fpvi(i, k)), ' sec']);

        % Proposed
        tic
        output_dense_cssd_pelt = cssd(x1, y1, p, gamma, [], delta);
        runtimes_dense_proposed_pelt(i, k) = toc;
        count_dense_proposed_pelt(i, k) = output_dense_cssd_pelt.complexity_counter;
        disp(['    RT Proposed + PELT: ', num2str(runtimes_dense_proposed_pelt(i, k)), ' sec']);
    end

    % EXPERIMENT 2: INCREASING NUMBER OF DISCONTINUITIES
    disp('  Ex 2')
    for i=1:length(signal_lengths)
        N = signal_lengths(i);
        disp(['    Processing N = ', num2str(N)]);
        x_s = sort(rand(N,1)*signal_lengths_mult(i));
        y2 = g(x_s - floor(x_s)) + sigma .* randn(N, 1);
        x2 = x_s/(signal_lengths_mult(i));
        delta = sigma * ones(N,1);
        
        % baseline
        output_rep_baseline = ruptures_cssd.detect_changepoints(np.array(x2), np.array(y2), p, gamma, np.array(delta));
        runtimes_rep_baseline(i, k) = output_rep_baseline{2};
        count_rep_baseline(i, k) = output_rep_baseline{4};
        disp(['    RT Baseline: ', num2str(runtimes_rep_baseline(i, k)), ' sec']);
        
        % proposed FPVI pruning
        tic
        output_rep_cssd_fpvi = cssd(x2, y2, p, gamma, [], delta, 'pruning', 'FPVI');
        runtimes_rep_proposed_fpvi(i, k) = toc;
        count_rep_proposed_fpvi(i, k) = output_rep_cssd_fpvi.complexity_counter;
        disp(['    RT Proposed + FPVI: ', num2str(runtimes_rep_proposed_fpvi(i, k)), ' sec']);

        % proposed PELT pruning
        tic
        output_rep_cssd_pelt = cssd(x2, y2, p, gamma, [], delta, 'pruning', 'PELT');
        runtimes_rep_proposed_pelt(i, k) = toc;
        count_rep_proposed_pelt(i, k) = output_rep_cssd_pelt.complexity_counter;
        disp(['    RT Proposed + PELT: ', num2str(runtimes_rep_proposed_pelt(i, k)), ' sec']);
    end
end

%% plot example signals
fig = figure(1); clf;
set(fig, 'Name', 'Sample signals', 'Color', 'white', 'Position', [0,0,1200,800]);
xx = linspace(0,1,30000);

subplot(2,1,1)
plot(x1, y1, 'ok')
hold on 
plot(xx, ppval(output_dense_cssd_fpvi.pp, xx), '.')
hold off
title('Example of type 1: Denser sampling')

subplot(2,1,2)
plot(x2, y2, 'ok')
hold on 
plot(xx, ppval(output_rep_cssd_fpvi.pp, xx), '.')
hold off
title('Example of type 2: Signal repetition')

%% Plot the results
runtimes_dense_proposed_fpvi_mean = mean(runtimes_dense_proposed_fpvi,2);
runtimes_rep_proposed_fpvi_mean = mean(runtimes_rep_proposed_fpvi,2);
runtimes_dense_proposed_pelt_mean = mean(runtimes_dense_proposed_pelt,2);
runtimes_rep_proposed_pelt_mean = mean(runtimes_rep_proposed_pelt,2);
runtimes_dense_baseline_mean = mean(runtimes_dense_baseline,2);
runtimes_rep_baseline_mean = mean(runtimes_rep_baseline,2);

count_dense_proposed_fpvi_mean = mean(count_dense_proposed_fpvi,2);
count_rep_proposed_fpvi_mean = mean(count_rep_proposed_fpvi,2);
count_dense_proposed_pelt_mean = mean(count_dense_proposed_pelt,2);
count_rep_proposed_pelt_mean = mean(count_rep_proposed_pelt,2);
count_dense_baseline_mean = mean(count_dense_baseline,2);
count_rep_baseline_mean = mean(count_rep_baseline,2);

fig = figure(2); clf;
set(fig, 'Name', 'CPU-Time', 'Color', 'white', 'Position', [0,0,800,400]);
subplot(2,2,1)
loglog(signal_lengths, runtimes_dense_baseline_mean, '-x', 'Linewidth', 2)
hold on
loglog(signal_lengths, runtimes_dense_proposed_fpvi_mean, '-x', 'Linewidth', 2)
loglog(signal_lengths, runtimes_dense_proposed_pelt_mean, '-x', 'Linewidth', 2)
hold off
title('Constant number of discont.')
legend({'Baseline algorithm', 'Proposed algorithm + FPVI', 'Proposed algorithm + PELT'}, 'Location', 'Northwest')
xlabel('Signal length')
ylabel('Runtime [sec]')
grid on

subplot(2,2,2)
loglog(signal_lengths, runtimes_rep_baseline_mean, '-x', 'Linewidth', 2)
hold on
loglog(signal_lengths, runtimes_rep_proposed_fpvi_mean, '-x', 'Linewidth', 2)
loglog(signal_lengths, runtimes_rep_proposed_pelt_mean, '-x', 'Linewidth', 2)
hold off
title('Linearly increasing number of discont.')
legend({'Baseline algorithm', 'Proposed algorithm + FPVI', 'Proposed algorithm + PELT'}, 'Location', 'Northwest')
xlabel('Signal length')
grid on

subplot(2,2,3)
loglog(signal_lengths, count_dense_baseline_mean, '-x', 'Linewidth', 2)
hold on
loglog(signal_lengths, count_dense_proposed_fpvi_mean, '-x', 'Linewidth', 2)
loglog(signal_lengths, count_dense_proposed_pelt_mean, '-x', 'Linewidth', 2)
hold off
title('Constant number of discont.')
legend({'Baseline algorithm', 'Proposed algorithm + FPVI', 'Proposed algorithm + PELT'}, 'Location', 'Northwest')
xlabel('Signal length')
ylabel('Counts data visited')
grid on

subplot(2,2,4)
loglog(signal_lengths, count_rep_baseline_mean, '-x', 'Linewidth', 2)
hold on
loglog(signal_lengths, count_rep_proposed_fpvi_mean, '-x', 'Linewidth', 2)
loglog(signal_lengths, count_rep_proposed_pelt_mean, '-x', 'Linewidth', 2)
hold off
title('Linearly increasing number of discont.')
legend({'Baseline algorithm', 'Proposed algorithm + FPVI', 'Proposed algorithm + PELT'}, 'Location', 'Northwest')
xlabel('Signal length')
grid on



