% Example: Old faithful data

%% load data
% Auto-generated by MATLAB
% Initialize variables.
filename = 'faithful.txt';
startRow = 15;
endRow = 286;

formatSpec = '%3f%10f%f%[^\n\r]';

% Open the text file.
fileID = fopen(filename,'r');

% Read columns of data according to the format.
textscan(fileID, '%[^\n\r]', startRow-1, 'WhiteSpace', '', 'ReturnOnError', false);
dataArray = textscan(fileID, formatSpec, endRow-startRow+1, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'ReturnOnError', false, 'EndOfLine', '\r\n');

% Close the text file.
fclose(fileID);

% Create output variable
faithful = table(dataArray{1:end-1}, 'VariableNames', {'VarName1','eruptions','waiting'});

% Clear temporary variables
clearvars filename startRow endRow formatSpec fileID dataArray ans;
%%% END LOAD DATA

% Preprocessing
% sort the data in ascending order
x_unsorted = faithful.eruptions;
y_unsorted = faithful.waiting;
[x, perm] = sort(x_unsorted);
y = y_unsorted(perm);
xx = linspace(min(x),max(x), 1000);

%% automatic parameter selection
% CSSD with 5-fold CV
rng(123)
% starting the optimization with these fine-tuned initial values turned out to 
% result in a reasonbly good CV score for this data
p_init = 0.56023; gamma_init = 817.7744;
output_cv = cssd_cv(x, y, [], [], [p_init; gamma_init], 'verbose', 1);

%% computing the results
p_cv = output_cv.p;
gamma_cv = output_cv.gamma;

p_1 = p_cv;
gamma_1 = gamma_cv;
output_1 = cssd(x, y, p_cv, gamma_cv);
cv_score_1 = output_cv.cv_fun(p_1, gamma_1);

gamma_2 = 145;
p_2 = p_cv;
output_2 = cssd(x, y, p_2, gamma_2);
cv_score_2 = output_cv.cv_fun(p_2, gamma_2);


gamma_3 = Inf;
p_3 = 0;
output_3 = cssd(x, y, p_3, gamma_3);
cv_score_3 = output_cv.cv_fun(p_3, gamma_3);

% plot the results
colors = colororder;
fig = figure(1); clf;
set(fig, 'Name', 'Geyser', 'Color', 'white', 'Position', [0 0 500 250]*1.2);
plot(x, y, 'ok', 'Markersize', 5, 'Linewidth', 1, 'DisplayName', 'Data')
hold on
output_1.pcw_fun.plot(xx, '-', 'Color', colors(3,:), 'Linewidth', 2, 'DisplayName', sprintf('CSSD with \\gamma = %.1f, p = %.5f, CV score: %.1f', gamma_1, p_1, cv_score_1));
output_2.pcw_fun.plot(xx, '--', 'Color', colors(5,:), 'Linewidth', 2, 'DisplayName', sprintf('CSSD with \\gamma = %.1f, p = %.5f, CV score: %.1f', gamma_2, p_2, cv_score_2));
output_3.pcw_fun.plot(xx, ':', 'Color', colors(7,:), 'Linewidth', 2, 'DisplayName', sprintf('Linear model, CV score: %.1f', cv_score_3));
xlabel('Duration of eruption (min)')
ylabel('Time to next eruption (min)')
for i = 1:numel(output_2.discont)
    plot([output_2.discont(i), output_2.discont(i)], ylim, '--', 'Color', '#999999', 'DisplayName','')
end
leg = legend('Location', 'SouthEast');
legend(leg.String{1:4})
hold off
box off