classdef PcwFunReal
    % PcwFunReal This class implements a piecewise defined function on the
    % real line. Its purpose is conveniently evaluating and plotting
    % piecewise functions, including vector-valued ones.
    %
    % B7 (audit): the original `eval` allocated `yy = NaN(size(xx))`, which
    % is correct only for scalar-output pieces. For vector-valued pieces
    % (csaps with dim > 1), `fun_cell{i}(xx_subset)` returns dim x N — so
    % the result must be shaped (numel(xx), dim). This class now returns a
    % consistent (numel(xx), dim) layout for any dim, including dim = 1.

    properties
        bounds
        fun_cell
        n_pieces
    end

    methods
        % Constructs piecewise function where the boundaries of the pieces
        % are stored in 'bounds' and the corresponding functions
        % are stored as a cell array of function handles in 'fun_cell'.
        function obj = PcwFunReal(bounds,fun_cell)
            obj.bounds = bounds;
            obj.n_pieces = numel(bounds) - 1;
            obj.fun_cell = fun_cell;
            assert(numel(fun_cell) == obj.n_pieces)
        end

        % Evaluates the piecewise defined function at the points in xx.
        % Output shape is (numel(xx), dim) where dim is the per-point output
        % dimension (1 for scalar pieces). At an interior boundary, the
        % midpoint of the left and right pieces is used.
        function yy = eval(obj, xx)
            xx = xx(:);                           % column for indexing
            % Probe one piece to discover dim. Use a finite point inside
            % the first piece (mid of bounds(1), bounds(2) clipped to xx
            % range) to avoid evaluating at a possible -Inf bound.
            probe_x = 0;
            if isfinite(obj.bounds(1))
                probe_x = obj.bounds(1) + 1;
            elseif isfinite(obj.bounds(2))
                probe_x = obj.bounds(2) - 1;
            end
            probe_y = obj.fun_cell{1}(probe_x);
            dim = numel(probe_y);
            yy = NaN(numel(xx), dim);
            for i = 1:obj.n_pieces
                idx = (obj.bounds(i) < xx) & (xx < obj.bounds(i+1));
                if any(idx)
                    v = obj.fun_cell{i}(xx(idx));
                    if dim > 1
                        % csaps/ppval returns dim x N for multi-dim pp;
                        % transpose to N x dim for assignment.
                        v = v.';
                    else
                        v = v(:);
                    end
                    yy(idx, :) = v;
                end
            end
            % If a point in xx coincides with an interior boundary, take
            % the midpoint of the left and right pieces.
            for i = 2:obj.n_pieces
                idx = (obj.bounds(i) == xx);
                if any(idx)
                    vl = obj.fun_cell{i-1}(xx(idx));
                    vr = obj.fun_cell{i}(xx(idx));
                    if dim > 1
                        vl = vl.'; vr = vr.';
                    else
                        vl = vl(:); vr = vr(:);
                    end
                    yy(idx, :) = 0.5 * (vl + vr);
                end
            end
            % Handle the endpoints (use the closest piece, no averaging).
            idx = (obj.bounds(1) == xx);
            if any(idx)
                v = obj.fun_cell{1}(obj.bounds(1));
                if dim > 1
                    yy(idx, :) = repmat(v(:).', sum(idx), 1);
                else
                    yy(idx, :) = v;
                end
            end
            idx = (obj.bounds(end) == xx);
            if any(idx)
                v = obj.fun_cell{end}(obj.bounds(end));
                if dim > 1
                    yy(idx, :) = repmat(v(:).', sum(idx), 1);
                else
                    yy(idx, :) = v;
                end
            end
        end

        function h = plot(obj, xx, varargin)
            xx_wob = xx(~ismember(xx, obj.bounds)); % clear bounds
            xx_wb = [xx_wob(:); obj.bounds(:)]; % add bounds
            yy_eval = obj.eval(xx_wob(:));
            % yy_eval is (numel(xx_wob), dim). Insert NaN rows for the
            % bound points so the plot draws as separated segments.
            dim = size(yy_eval, 2);
            yy_wb = [yy_eval; nan(numel(obj.bounds), dim)];
            [xx_plot, perm] = sort(xx_wb);
            yy_plot = yy_wb(perm, :);
            h = plot(xx_plot, yy_plot, varargin{:});
        end
    end
end
