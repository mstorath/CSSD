function pp = embed_pptocubic(pp)
%embed_pptocubic embeds a pp of order lower than 4 to a cubic pp 
%so that is has exactly 4 coefficients. (The higher order coefficients are
%filled with zeros)
if pp.order < 4
    [m,~] = size(pp.coefs);
    pp.coefs = [zeros(m, 4-pp.order), pp.coefs];
    pp.order = 4;
end

end

