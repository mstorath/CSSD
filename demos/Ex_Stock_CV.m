% Example: Stock data

% BEGIN DATA SOURCE 
% https://www.macrotrends.net/stocks/charts/FB/meta-platforms/stock-price-history
% "MacroTrends Data Download"
% "FB - Historical Price and Volume Data"
% "Note: Historical prices are adjusted for stock splits."
%
% "Disclaimer and Terms of Use: Historical stock data is provided 'as is' and solely for informational purposes, not for trading purposes or advice."
% "MacroTrends LLC expressly disclaims the accuracy, adequacy, or completeness of any data and shall not be liable for any errors, omissions or other defects in, "
% "delays or interruptions in such data, or for any actions taken in reliance thereon.  Neither MacroTrends LLC nor any of our information providers will be liable"
% "for any damages relating to your use of the data provided."
%
%
% "ATTRIBUTION: Proper attribution requires clear indication of the data source as ""www.macrotrends.net""."
% "A ""dofollow"" backlink to the originating page is also required if the data is displayed on a web page."
% END DATA SOURCE

%% data preparation
load Table_MetaStock.mat

N = numel(Table_MetaStock.close);
dates = cell(N,1);
price = zeros(N,1);
for i=1:N
    dates{i} = datestr(Table_MetaStock.date(i));
    price(i) = Table_MetaStock.close(i);
end
y = log(price');
N = numel(y);
x = daysact(dates{1}, cell2mat(dates));
dates_ext = datetime(dates{1}):datetime(dates{end});

%% determine parameters by cross validation
rng(123)

% starting the optimization with these fine-tuned initial values turned out to 
% result in a reasonbly good CV score for this data
p_init = 0.4702;  
gamma_init = 0.0069;
output_cv = cssd_cv(x, y, [], [], [p_init; gamma_init], 'verbose', 1);

%% cssd estimation
p = output_cv.p;
gamma = output_cv.gamma;
output = cssd(x, y, p, gamma);
discont = output.discont;
pp_cssd = output.pp;

%% plot result
xx = linspace(min(x), max(x),100000);
fig = figure(1);
clf;
set(gcf, 'name', 'Stock', 'Color', 'white', 'units','normalized', 'position', [0,0,0.3,0.3]);
plot([],[])
hold on
output.pcw_fun.plot(xx, '-', 'Linewidth', 2, 'Color', '#77AC30', 'DisplayName', 'CSSD')
plot(x, y, '.k', 'Markersize', 0.5, 'DisplayName', 'Data')
legend('Location', 'NorthWest')
xlim([min(xx), max(xx)])
for i = 1:numel(discont)
    plot([discont(i), discont(i)], ylim, '--', 'Color', '#999999', 'DisplayName','')
end
xticks(discont);
xtickangle(60)
dates_ext_str = datestr(dates_ext);
xticklabels(dates_ext_str(round(xticks),:));
hold off
leg = legend;
legend(leg.String{1:3})