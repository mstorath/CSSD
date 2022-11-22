classdef PcwFunReal
    % PcwFunReal This class implements a piecewise define functions on the
    % real line. Its purpose is conveniently plotting piecewise functions
   
    properties
        bounds
        fun_cell
        n_pieces
    end
    
    methods
        % Constructs piecewise function where the boundaries of the pieces
        % are stored in 'bounds' and the corresponding functions 
        % and the corresponding functions are stored as cell array of function handles
        % in 'fun_cell'
        function obj = PcwFunReal(bounds,fun_cell)
            obj.bounds = bounds;
            obj.n_pieces = numel(bounds) - 1;
            obj.fun_cell = fun_cell;
            assert(numel(fun_cell) == obj.n_pieces)
        end
        
        % Evaluates the piecewise defined function at points given in xx.
        % If xx(i) coincides with a boundary, the midpoint of the left and
        % right function is taken
        function yy = eval(obj, xx)
            yy = NaN(size(xx));
            for i=1:obj.n_pieces
                idx = find( (obj.bounds(i) < xx) & (xx < obj.bounds(i+1)) );
                yy(idx) = obj.fun_cell{i}(xx(idx));
            end
            % If a point in xx coincides with a boundary which are not endpoints
            % the midpoint of the left and
            % right function is taken
            for i=2:obj.n_pieces 
                idx = find( obj.bounds(i) == xx);
                yy(idx) = 0.5 * (obj.fun_cell{i-1}(xx(idx)) + obj.fun_cell{i}(xx(idx)));
            end
            % handling the endpoints
            idx = find( obj.bounds(1) == xx);
            yy(idx) = obj.fun_cell{1}(obj.bounds(1));
            idx = find( obj.bounds(end) == xx);
            yy(idx) = obj.fun_cell{end}(obj.bounds(end));
        end

        function h = plot(obj, xx, varargin)
            xx_wob = xx(~ismember(xx, obj.bounds)); % clear bounds
            xx_wb = [xx_wob(:); obj.bounds(:)]; % add bounds
            yy_wb = [obj.eval(xx_wob(:)); nan(size(obj.bounds(:)))];
            [xx_plot, perm] = sort(xx_wb);
            yy_plot = yy_wb(perm);
            plot(xx_plot,yy_plot, varargin{:});
        end
    end
end

