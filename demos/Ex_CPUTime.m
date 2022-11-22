% Example: Runtime for different types of signals

% HeaviSine signal
g = @(x) 4.*sin(4*pi.*x) - sign(x - .3) - sign(.72 - x);

% parameters
p = 0.9999;
gamma = 20;
sigma = 0.4;

% signal lengths
signal_lengths_mult = [1,2,4,8,16,32];
signal_lengths = signal_lengths_mult*200;

% number of runs
K = 5;

%
runtimes_dense = zeros(numel(signal_lengths), K);
runtimes_rep = zeros(numel(signal_lengths), K);



for k = 1:K
    disp(['Run = ', num2str(k), ' of ', num2str(K)])
    % EXPERIMENT 1: DENSER SAMPLING, SAME NUMBER OF DISCONTINUITIES
    disp('  Ex 1')
    for i=1:length(signal_lengths)
        N = signal_lengths(i);
        disp(['    Processing N = ', num2str(N)]);
        x1 = (1:N)'/N;
        y1 = g(x1) + sigma .* randn(N, 1);
        delta = sigma * ones(N,1);
        tic
        output_cssd1 = cssd(x1, y1, p, gamma, [], delta);
        runtimes_dense(i, k) = toc;
        disp(['    RT: ', num2str(runtimes_dense(i, k)), ' sec']);
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
        tic
        output_cssd2 = cssd(x2, y2, p, gamma, [], delta);
        runtimes_rep(i,k) = toc;
        disp(['    RT: ', num2str(runtimes_rep(i, k)), ' sec']);
    end
end

%% plot example signals
fig = figure(1); clf;
set(fig, 'Name', 'Sample signals', 'Color', 'white', 'Position', [0,0,1200,800]);
xx = linspace(0,1,30000);

subplot(2,1,1)
plot(x1, y1, 'ok')
hold on 
plot(xx, ppval(output_cssd1.pp, xx), '.')
hold off
title('Example of type 1: Denser sampling')

subplot(2,1,2)
plot(x2, y2, 'ok')
hold on 
plot(xx, ppval(output_cssd2.pp, xx), '.')
hold off
title('Example of type 2: Signal repetition')

%% Plot the results
runtimes_dense_mean = mean(runtimes_dense,2);
runtimes_rep_mean = mean(runtimes_rep,2);

fig = figure(2); clf;
set(fig, 'Name', 'CPU-Time', 'Color', 'white', 'Position', [0,0,800,400]);
loglog(signal_lengths, runtimes_dense_mean, '-x', 'Linewidth', 2)
hold on
loglog(signal_lengths, runtimes_rep_mean, '-x', 'Linewidth', 2)
hold off
legend({'Constant number of discont.', 'Linearly increasing number of discont.'}, 'Location', 'Northwest')
xlabel('Signal length')
ylabel('Runtime [sec]')
grid on