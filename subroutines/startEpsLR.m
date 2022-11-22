function [eps_lr, R, z] = startEpsLR(y, d, alpha, beta)
%startEpsLR Initializes the fast computation of the spline energies 

% create design matrix
A = zeros(4,4);
A(1,1) = alpha(1);
A(2:3,:) = beta * [2.*3.^(1/2).*d.^(-3/2),3.^(1/2).*d.^(-1/2),(-2).*3.^(1/2).*d.^(-3/2),3.^(1/2).*d.^(-1/2);...
    0,d.^(-1/2),0,(-1).*d.^(-1/2)];
A(4,3) = alpha(2);

% create rhs
z = zeros(4, size(y, 2));
z(1,:) = alpha(1) * y(1,:);
z(4,:) = alpha(2) * y(2,:);
%z =  [alpha(1) * y(:,1)'; zeros(size(y,2)); 0; alpha(2) * y(:,2)'];

% perform lsq solution by QR and store current state
[Q,R] = qr(A);
z = Q' * z;
eps_lr = 0;
R = [R; zeros(1,4)]; % for compatibility with update
z = [z; zeros(1,size(z,2))];
end

