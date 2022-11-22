% Example: Synthetic test signal

% random seed for reproducibility
rng(123)

% HeaviSine signal
g1 = @(x) 4*(besselj(1, 20 * x) + x .* ((0.3) <= x) .* (x <= 0.4) - x .* ((0.6) <= x) .* (x <= 1));
g2 = @(x) 4.*sin(4*pi.*x) - sign(x - .3) - sign(.72 - x);

% setup computation
N = 200;
p_cell = {0.9999};
gamma_cell = {13, 15, 17,  Inf};
I = numel(p_cell);
J = numel(gamma_cell);
K = 1000;
pp_cssd_cell = cell(I,J,K);
discont_cell = cell(I,J,K);
x_cell = cell(K,1);
y_cell = cell(K,1);
sigma = 0.6;
delta = sigma * ones(N, 1);

for k=1:K
    % random sample points
    x = sort(rand(N,1));
    % noisy observations
    y = [g1(x) + sigma .* randn(N, 1), g2(x) + sigma .* randn(N, 1)];
    % compute the splines and store them in cell arrays
    x_cell{k} = x;
    y_cell{k} = y;
    for i = 1:I
        for j = 1:J
            p = p_cell{i};
            gamma = gamma_cell{j};
            output = cssd(x, y', p, gamma, [], delta);
            pp_cssd_cell{i,j,k} = output.pp;
            discont_cell{i,j,k} = output.discont;

        end
    end
end

%% Create the figure
nn = 5000;
xx = linspace(0,1, nn);

fig = figure(1); clf;
set(fig, 'Name', 'Synthetic signal', 'Color', 'white', 'Position', [0 0 800 700] * 0.8);
T = tiledlayout(18,1,'TileSpacing','tight','Padding','compact');
ylim_common = [-8,6];

colors = {'#77AC30', '#4DBEEE'};
markers = {'o', 'x'};

true_discont = [0.3, 0.4, 0.6, 0.72];
x_ticks = [0, true_discont, 1];

t=tiledlayout(T,1,2, 'TileSpacing','compact','Padding','compact');
t.Layout.Tile=1;
t.Layout.TileSpan=[4,1];
%
nexttile(t)
yy_true = [g1(xx)'  g2(xx)'];
plot([],[])
hold on
for v=1:2
    plot(xx, yy_true(:,v), '.', 'DisplayName', 'True signal', 'Color', colors{v})
end
hold off
ylim(ylim_common);
title('(a) True signal')
box off
xticks(x_ticks)
hold on
plot(x_ticks, ones(size(x_ticks)) * ylim_common(1), '|k')
hold off

%
nexttile(t)
plot([],[])
hold on
for v=1:2
    plot(x_cell{1}, y_cell{1}(:,v), markers{v}, 'Markersize', 5, 'Linewidth', 1, 'DisplayName', 'Data', 'Color', colors{v})
end
hold off
ylim(ylim_common);
title('(b) Sample realization')
box off
xticks(x_ticks)
hold on
plot(x_ticks, ones(size(x_ticks)) * ylim_common(1), '|k')
hold off

%
i=1;
s = [6, 13];
title_labels = {'(c)', '(d)', '(e)', '(f)'};
for r=1:2
    t=tiledlayout(T,1,2, 'TileSpacing','compact','Padding','compact');
    t.Layout.Tile=s(r);
    t.Layout.TileSpan=[4,1];
    for j = (1:2) + (r-1) * 2

        p = p_cell{i};
        gamma = gamma_cell{j};
        yy_cssd = zeros(K,nn,2);
        for k=1:K
            yy_cssd(k,:,:) = ppval(pp_cssd_cell{i,j,k}, xx)';
        end
        nexttile(t)

        plot([],[]);
        hold on
        for v=1:2
            plot(xx, yy_cssd(1,:,v), '.', 'Color', colors{v})
            xconf = [xx, xx(end:-1:1)];
            yconf = [quantile(yy_cssd(:,:,v),0.025,1), fliplr(quantile(yy_cssd(:,:,v),0.975,1))];
            fill(xconf,yconf,'red', 'Edgecolor', '#77AC30', 'FaceColor', colors{v}, 'FaceAlpha', 0.3);
            ylim(ylim_common)
            xticks(x_ticks)
        end
        hold off
        hold on
            plot(x_ticks, ones(size(x_ticks)) * ylim_common(1), '|k')
            hold off
        if gamma == Inf
            title([ title_labels{j}, ' Smoothing spline (equiv. to CSSD with \gamma = \infty)'])
        else
            title([title_labels{j}, ' CSSD, \gamma = ', num2str(gamma)])
        end
    end
    t=tiledlayout(T,1,2, 'TileSpacing','compact','Padding','compact');
    t.Layout.Tile=s(r)+4;
    t.Layout.TileSpan=[2,1];
    for j = (1:2) + (r-1) * 2
        nexttile(t)
        discont_all = [];
        for k=1:K
            discont_all = [discont_all; discont_cell{i,j,k}];
        end
        if numel(discont_all) > 0
            edges = linspace(-0.005, 1.005, 102);
            histogram(discont_all, edges, 'FaceColor', '#77AC30');
            xlim([0,1]);
            ylim([0,K])
            box off
            xticks(x_ticks)
            hold on
            plot(x_ticks, zeros(size(x_ticks)), '|k')
            hold off
        else
            axis off
        end
    end
end