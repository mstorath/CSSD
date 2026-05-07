function enSmooth = spline_innerenergy(pp)
%spline_innerenergy
% Computes the inner energy of a cubic spline (C^2 continuous!) in pp-form
% Input: Cubic smoothing spline pp (must be C^2 continuous to give correct results)
% Output: Inner energy of the cubic spline \int_{-\inf}^{\inf} (pp''(x))^2 dx
%
% N4 (audit): the formula evaluates pp'' at piece boundaries via ppval,
% which picks one side of the discontinuity at the break between the
% original cubic region and the linear extension introduced by fnxtr.
% This gives the correct integral *only* when pp''(x_1) = pp''(x_N) = 0
% — i.e. for natural cubic splines such as those returned by csaps.
% Passing a non-natural spline (e.g. one with arbitrary boundary
% conditions) produces a subtly wrong answer at the boundaries.

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

