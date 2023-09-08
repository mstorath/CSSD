function pp = linext_pp(pp, l, r)
%LINEXTPP Extends a cubic spline in ppform linearly beyond its boundaries
%to the boundaries [l, r]

assert( (l <= pp.breaks(1)) && (pp.breaks(end) <= r), 'New boundaries must be larger than the old ones for extension.')
pp = embed_pptocubic(pp);

pp_deriv = pp;
pp_deriv.coefs = pp_deriv.coefs(:, 1:3) .* [3,2,1];
pp_deriv.order = 3;
first = pp.breaks(1);
last  = pp.breaks(end);

new_breaks = [l, pp.breaks, r];
base_last = ppval(pp, last);

slope_last = ppval(pp_deriv, last);

base_first = ppval(pp, first);
slope_first = ppval(pp_deriv, first);
base_l = base_first + slope_first .* (l - first);

new_coefs  = [zeros(pp.dim, 2), slope_first, base_l; pp.coefs; zeros(pp.dim, 2), slope_last, base_last];

pp.breaks = new_breaks;
pp.coefs = new_coefs;
pp.pieces = pp.pieces + 2;

end

