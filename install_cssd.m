% sets Matlab paths for CSSD toolbox
function install_cssd
fprintf('Adding CSSD paths to Matlab path...\n')
folder = fileparts(which(mfilename('fullpath')));
addpath(genpath(folder));
savepath;
try
    % B3 (audit): smoke test uses N=3 instead of N=2 to avoid the
    % rcv_score / (N-2) divide-by-zero (N=2 was producing a NaN field).
    cssd(1:3, [0 1 0], 1, 1);
    fprintf('Done.\n')
catch ME
    if strcmp(ME.identifier, 'MATLAB:UndefinedFunction')
        fprintf(['Curve Fitting Toolbox appears to be missing.\n' ...
                 'CSSD depends on `csaps`, `fnxtr`, `fnder`, `ppmak`, ' ...
                 '`ppval`, and the private `chckxywp` from that toolbox.\n' ...
                 'Please install it and rerun this script.\n']);
    else
        fprintf(['Setting path automatically failed: %s\n' ...
                 'Try adding the path to the CSSD folder and the ' ...
                 'subfolders manually.\n'], ME.message);
    end
end

end
