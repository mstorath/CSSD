function pp_merged = merge_ppcell(pp_cell)
% merge_ppcell Merges a cell array of piecewise polynomials in pp form 
% with matching endpoints into a single pp
    breakpoints = zeros(numel(pp_cell)-1,1);
    pp_merged = pp_cell{1};
    for i = 2:numel(pp_cell)
        pp = pp_cell{i};
        breakpoints(i-1) = pp_merged.breaks(end);
        pp_merged.breaks = [pp_merged.breaks(1:end-1), pp.breaks];
        pp_merged.coefs = [pp_merged.coefs; pp.coefs];
        pp_merged.pieces = numel(pp_merged.breaks) -1;
    end
end
