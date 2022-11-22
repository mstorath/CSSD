function pp = linext_pp(pp, l, r)
%LINEXTPP Extends a cubic spline in ppform linearly beyond its boundaries
%to the boundaries [l, r]

assert( (l <= pp.breaks(1)) && (pp.breaks(end) <= r), 'New boundaries must be larger than the old ones for extension.')

% extends the boundaries by one in each direction and continues the spline
% linearly (fnxtr is Matlab built-in)
pp = fnxtr(pp,2);

% adjust the right bound 
pp.breaks(end) = r;

% when changing the first break, the base point of the polynomial is changed
% which requires a corrected     
for i = 1:pp.dim % loop over all dimensions
    a = pp.coefs(i,end-1);
    pp.coefs(i,end) = pp.coefs(i,end) - a * (pp.breaks(1) - l);
    % finally adjust the endpoints
end
pp.breaks(1) = l;

end

