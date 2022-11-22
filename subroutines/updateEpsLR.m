function [eps_lrp1, R_rp1, z_rp1] = updateEpsLR(eps_lr, R_r, y_rp1, d_r, z_r, alpha, beta)
%updateEpsLR Updates the spline energy on the interval 1:r+1 from the
%state on the interval 1:r

% create design matrix
R_rp1 = zeros(5,4);
R_rp1(1:2,1:2) = R_r(3:4,3:4);
R_rp1(3:4, :) = beta * [2.*3.^(1/2).*d_r.^(-3/2),3.^(1/2).*d_r.^(-1/2),(-2).*3.^(1/2).*d_r.^( ...
    -3/2),3.^(1/2).*d_r.^(-1/2);...
    0,d_r.^(-1/2),0,(-1).*d_r.^(-1/2)];
R_rp1(5,:) = [0,0,alpha,0];

% create rhs
z_rp1 = zeros(5, size(y_rp1, 2));
z_rp1(1:2,:) = z_r(3:4,:);
z_rp1(5,:) =  alpha*y_rp1;

% perform lsq solution by QR and store current state
[Q,R_rp1] = qr(R_rp1);
z_rp1 = Q' * z_rp1;
eps_lrp1 = eps_lr + sum(z_rp1(5,:).^2);

end

