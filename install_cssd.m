% sets Matlab paths for CSSD toolbox
function install_cssd
fprintf('Adding CSSD paths to Matlab path...\n')
folder = fileparts(which(mfilename('fullpath')));
addpath(genpath(folder));
savepath;
try
    cssd([0,1],[0,0], 1, 1);
    fprintf('Done.\n')
catch
    fprintf(['Setting path automatically failed. \n' ...
        'Try to add the path to the CSSD folder and the subfolders manually.\n'])
end

end