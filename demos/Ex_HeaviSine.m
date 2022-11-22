% Example: HeaviSine test signal

% random seed for reproducibility
rng(123)

% HeaviSine signal
g = @(x) 4.*sin(4*pi.*x) - sign(x - .3) - sign(.72 - x);

% setup computation
N = 200;
p_cell = {0.9999};
gamma_cell = {10, 20, 30, Inf};
I = numel(p_cell);
J = numel(gamma_cell);
K = 1000;
pp_cssd_cell = cell(I,J,K);
discont_cell = cell(I,J,K);
x_cell = cell(I,J,K);
y_cell = cell(I,J,K);
sigma = 0.4;
delta = sigma * ones(N, 1);

for k=1:K
    % random sample points
    x = sort(rand(N,1));
    % noisy observations
    y = g(x) + sigma .* randn(N, 1);
    % compute the splines and store them in cell arrays
    for i = 1:I
        for j = 1:J
            p = p_cell{i};
            gamma = gamma_cell{j};
            output = cssd(x, y, p, gamma, [], delta);
            pp_cssd_cell{i,j,k} = output.pp;
            discont_cell{i,j,k} = output.discont;
            x_cell{i,j,k} = x;
            y_cell{i,j,k} = y;
        end
    end
end

%% Create the figure
nn = 5000;
xx = linspace(0,1, nn);
fig = figure(1); clf;
set(fig, 'Name', 'HeaviSine', 'Color', 'white', 'Position', [0 0 800 700]);
T = tiledlayout(18,1,'TileSpacing','tight','Padding','compact');
ylim_common = [-8,6];

t=tiledlayout(T,1,2, 'TileSpacing','compact','Padding','compact'); 
t.Layout.Tile=1;
t.Layout.TileSpan=[4,1];
%
nexttile(t)
yy_true = g(xx);
plot(xx, yy_true, '.', 'DisplayName', 'True signal', 'Color', '#0072BD')
ylim(ylim_common);
title('(a) True signal')
box off

%
nexttile(t)
plot(x_cell{1}, y_cell{1}, 'ok', 'Markersize', 5, 'Linewidth', 1, 'DisplayName', 'Data')
ylim(ylim_common);
title('(b) Sample realization')
box off

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
        yy_cssd = zeros(K,nn);
        for k=1:K
            yy_cssd(k,:) = ppval(pp_cssd_cell{i,j,k}, xx);
        end
        nexttile(t)
        plot(xx, yy_cssd(1,:), '.', 'Color', '#77AC30')
        xconf = [xx, xx(end:-1:1)];
        yconf = [quantile(yy_cssd,0.025,1), fliplr(quantile(yy_cssd,0.975,1))];
        hold on
        fill(xconf,yconf,'red', 'Edgecolor', '#77AC30', 'FaceColor', '#77AC30', 'FaceAlpha', 0.3);
        hold off
        box off
        ylim(ylim_common)
       
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
        else
            axis off
        end
    end
end
