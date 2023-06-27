% Example: Synthetic test signal and comparison
% with Bayesian ensemble method (beast toolbox of
% https://github.com/zhaokg/Rbeast)

% random seed
rng(123)
% synthetic signal
g = @(x) besselj(1, 20 * x) + x .* ((0.3) <= x) .* (x <= 0.4) - x .* ((0.6) <= x) .* (x <= 1);

% setup computation
N = 100;
K = 100;
sigma_arr = [0.05, 0.1, 0.15];
J = numel(sigma_arr);
x_cell = cell(K,J);
y_cell = cell(K,J);

% generate synthetic signal
for k=1:K
    for j=1:J
        % random sample points
        x = sort(rand(N,1));
        % noisy observations
        y = g(x) + sigma_arr(j) .* randn(N, 1);
        % compute the data and store them in cell arrays
        x_cell{k,j} = x;
        y_cell{k,j} = y;
    end
end

% setup outputs
n_folds = 5;
pp_cell_cssd = cell(K,J);
discont_cell_cssd = cell(K,J);
pp_cell_beast = cell(K,J);
discont_beast_cell = cell(K,J);
p_arr = zeros(K,J);
gamma_arr = zeros(K,J);
runtimes = zeros(K,J);
out_cssd = cell(K,J);
out_beast = cell(K,J);



% load rbeast toolbox
eval(webread('http://b.link/rbeast',weboptions('cert','')))
beast_deltat = 1/100;

% perform computations for all signals
for k=1:K
    fprintf('Processing number %i of %i...', k, K)
    for j=1:J
        x = x_cell{k,j};
        y = y_cell{k,j};
        delta = sigma_arr(j) * ones(N, 1);
        % Compute CSSD result
        tic
        % use starting point p = 0.99 , gamma = 1
        output_cv = cssd_cv(x, y, [], [],  delta, [0.99; 1]);
        p_arr(k,j) = output_cv.p;
        gamma_arr(k,j) = output_cv.gamma;
        out_cssd{k,j} = cssd(x, y, output_cv.p, output_cv.gamma, [], delta);
        runtimes(k,j) = toc;
        % Compute beast result
        out_beast{k,j}  = beast_irreg(y', 'time', x', 'season', 'none', 'deltat', beast_deltat);  % trend-only data without seasonality
    end
    fprintf('Elapsed time: %f', sum(runtimes(k,:)))
    fprintf(', estimated remaining time in hours: %f\n', sum(mean(runtimes(1:k,:))) * (K - k)/3600 )
end


%%
nn = 5000;
xx = linspace(0,1, nn);
clf
fig = figure(1); clf;
set(fig, 'Name', 'Synthetic signal CV', 'Color', 'white', 'Position', [0 0 800 900]*0.8);
T = tiledlayout(15,3,'TileSpacing','tight','Padding','compact');
ylim_common = [-2,1.1];

% vector of true discontinuities and corresponding xticks
true_discont = [0.3, 0.4, 0.6];
x_ticks = [0, true_discont, 1];
% selection of representative samples 
sample_ks = [10, 75, 23]; 
% plot data
for j=1:J
    nexttile([3,1])

    x = x_cell{sample_ks(j),j};
    y = y_cell{sample_ks(j),j};
    plot(x, y, 'ok', 'Markersize', 5, 'Linewidth', 1, 'DisplayName', 'Data')
    box off
    ylim(ylim_common)
    xlim([0,1]);
    xticks(x_ticks)
    hold on
    plot(x_ticks, ones(size(x_ticks)) * ylim_common(1), '|k')
    hold off
    if j==2
        title(sprintf('(a) Sample realizations for noise levels \\sigma = %.2f, %.2f, %.2f',sigma_arr))
    end

end
% empty plots for distance
for j=1:J
    nexttile([1,1])
    axis off
end

colors = get(gca, 'ColorOrder');

% plot two stage
for j=1:J
    nexttile([3,1])
    yy_beast = zeros(K,nn);
    for k=1:K
        out = out_beast{k,j};
        x_beast = out.time;
        x_beast_data = x_beast(~isnan(out.data));
        y_beast = out.trend.Y;
        y_beast_data = y_beast(~isnan(out.data));
        % use median statistics as used by default in plotbeast
        cp_rhs = out.trend.cp(1:out.trend.ncp_median);
        % correction of cp because beast considers discontinuities at right bin edge instead of midpoint
        cp_rhs = unique(sort(cp_rhs(~isnan(cp_rhs))));
        cp_lhs = zeros(size(cp_rhs));
        for i=1:numel(cp_rhs)
            cp_dist = x_beast_data - cp_rhs(i);
            cp_dist = cp_dist(cp_dist < -beast_deltat/2);
            cp_lhs(i) = max(cp_dist) + cp_rhs(i);
        end
        discont_beast = (cp_rhs + cp_lhs)/2;
        discont_beast_cell{k,j} = discont_beast ;
        bounds = unique([0; discont_beast; 1]);
        % create pcw spline from bounds
        fun_cell = cell(numel(bounds)-1,1);
        for t=1:numel(bounds)-1
            idx = find((bounds(t) <= x_beast_data) & (x_beast_data < bounds(t+1)));
            if numel(idx) == 1
                pp = ppmak([-Inf, Inf], y_beast_data(idx));
            else
                pp= csape(x_beast_data(idx), y_beast_data(idx), 'variational');
                pp = fnxtr(pp,2);
            end
            fun_cell{t} = @(xx) ppval(pp, xx);
        end
        pcw = PcwFunReal(bounds, fun_cell);
        yy_beast(k,:) = pcw.eval(xx);
    end
    plot(xx, yy_beast(sample_ks(j),:), '.', 'Color', colors(1,:))
    xconf = [xx, xx(end:-1:1)];
    yconf = [quantile(yy_beast,0.025,1), fliplr(quantile(yy_beast,0.975,1))];
    hold on
    fill(xconf,yconf,'red', 'Edgecolor', colors(1,:), 'FaceColor', colors(1,:), 'FaceAlpha', 0.3);
    xticks(x_ticks)
    for d=true_discont
        plot(d * ones(2,1), ylim_common, '--', 'Color', '#999999')
    end
    hold off
    box off
    xlim([0,1]);
    
    ylim(ylim_common)
    if j==2
        title('(b) Baseline method based on the beast toolbox')
    end
end
for j=1:J
    nexttile([2,1])
    discont_all_beast = [];
    for k=1:K
        discont_beast = discont_beast_cell{k,j};
        discont_all_beast = [discont_all_beast; discont_beast];
    end
    if numel(discont_all_beast) > 0
        edges = linspace(-0.005, 1.005, 102);
        histogram(discont_all_beast, edges, 'FaceColor', colors(1,:));
        xlim([0,1]);
        ylim([0,K])
        xticks(x_ticks)
        hold on
        plot(x_ticks, zeros(size(x_ticks)), '|k')
        hold off
        box off
    else
    end
end
% empty plots for distance
for j=1:J
    nexttile([1,1])
    axis off
end

% plot CSSD
for j=1:J
    nexttile([3,1])

    yy_cssd = zeros(K,nn);
    for k=1:K
        yy_cssd(k,:) = ppval(out_cssd{k,j}.pp, xx);
    end
    plot(xx, yy_cssd(sample_ks(j),:), '.', 'Color', '#77AC30')
    xconf = [xx, xx(end:-1:1)];
    yconf = [quantile(yy_cssd,0.025,1), fliplr(quantile(yy_cssd,0.975,1))];
    hold on
    fill(xconf,yconf,'red', 'Edgecolor', '#77AC30', 'FaceColor', '#77AC30', 'FaceAlpha', 0.3);
    xticks(x_ticks)
    for d=true_discont
        plot(d * ones(2,1), ylim_common, '--', 'Color', '#999999')
    end
    hold off
    box off
    ylim(ylim_common)
    xlim([0,1]);
    if j==2
        title('(c) CSSD results, parameter selected by K-fold CV')
    end
end

for j=1:J
    nexttile([2,1])
    discont_all_cssd = [];
    for k=1:K
        discont_all_cssd = [discont_all_cssd; out_cssd{k,j}.discont];
    end
    if numel(discont_all_cssd) > 0
        edges = linspace(-0.005, 1.005, 102);
        histogram(discont_all_cssd, edges, 'FaceColor', '#77AC30');
        xlim([0,1]);
        ylim([0,K])
        xticks(x_ticks)
        hold on
        plot(x_ticks, zeros(size(x_ticks)), '|k')
        hold off
        box off
    else
        axis off
    end

end
