function pp_merged = merge_ppcell(pp_cell)
% merge_ppcell Merges a cell array of piecewise polynomials in pp form
% with matching endpoints into a single pp.
%
% N3 (audit): assert that consecutive pp's have matching endpoints. The
% original silently dropped the previous-end break when it differed from
% the next-start break, evaluating the previous piece over a slightly
% wrong t-range.

    pp_merged = pp_cell{1};
    for i = 2:numel(pp_cell)
        pp = pp_cell{i};
        assert(abs(pp_merged.breaks(end) - pp.breaks(1)) < 1e-12, ...
            'merge_ppcell:EndpointMismatch', ...
            'pp_cell{%d} starts at %g but pp_cell{%d} ends at %g.', ...
            i, pp.breaks(1), i-1, pp_merged.breaks(end));
        pp_merged.breaks = [pp_merged.breaks(1:end-1), pp.breaks];
        pp_merged.coefs = [pp_merged.coefs; pp.coefs];
        pp_merged.pieces = numel(pp_merged.breaks) - 1;
    end
end
