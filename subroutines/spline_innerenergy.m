function enSmooth = spline_innerenergy(pp)
%spline_innerenergy 
% Computes the inner energy of a cubic spline (C^2 continuous!) in pp-form
% Input: Cubic smoothing spline pp (must be C^2 continuous to give correct results)
% Output: Inner energy of the cubic spline \int_{-\inf}^{\inf} (pp''(x))^2 dx

% extend the spline linearly beyond boundaries
pp = fnxtr(pp,2);
% compute second derivative
ddpp = fnder(fnder(pp));
% compute integral \int_{-\inf}^{\inf} (pp''(x))^2 dx 
h = diff(ddpp.breaks);
l0 = ppval(ddpp, ddpp.breaks(1:end-1));
lh = ppval(ddpp, ddpp.breaks(2:end));
enSmooth = sum( h .* (l0.^2 + l0.*lh + lh.^2))/3;
end

