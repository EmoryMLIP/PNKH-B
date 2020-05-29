function [z, resid] = interiorQPB(H,Hinv,y,low,up,varargin)
% function [z, resid] = interiorQPB(H,Hinv,y,low,up,varargin)
%
% interior point method for the solving projection problem
%
% (1)  min_z 0.5*|z-y|^2_H subject to low <= z <= up
%
% where H is a spd matrix. The problem can also be written as
%
% (2)  min_z 0.5*z'*H*z - z'*H*y subject to low <= z <= up.
%
% The performance of the method hinges upon having a fast way to solve linear
% systems like
%
% (3) (H + diag(s)) dz = rhs
%
% For some vector s in the non-negative orthant. Therefore, we assume that
% the user provides a function handle Hinv, such that dz = Hinv(rhs,s)
% solves (3).
%
% Required Inputs:
%
%  H    - function handle for computing H(v,s) = (H+diag(s))*v
%  Hinv - function handle for solving (3) given rhs and s; H(v,s) = (H+diag(s))\v
%  y    - point to be projected, vector
%  low  - vector of lower bounds, put a large negative number if no bound
%  up   - vector of upper bounds, put a large positive number if no bound
%
% Optional Inputs (configured via varargin, see code for defaults)
%
% sigma   - central path parameter
% tau     - step size paramter, see sec 4.2  or (16.66) in Nocedal/Wright
% c       - shift of Hessian
% tol     - tolerance for ||Rprime||_2 and ||Rdual||_2
% maxIter - maximum number of iterations
% z       - starting guess for z, low <= z <= high
% w       - starting guess for slack variables, w > 0
% lambda  - starting guess for Lagrange multiplier
% out     = printing option, (0 -> no printing, 1 -> print final status, 2 -> print each iteration)

if nargin==0
    % test this function with the Matlab built-in quadprog
    n = 1000;
    m = 100;
    rng(1)
    A = randn(m,n);
    Hm = A'*A + eye(n);
    low = 0.5*ones(n,1);
    up = ones(n,1);
    y  = 10*(rand(n,1)-5);
    
    H    = @(v,s) (Hm + diag(s))*v;
    Hinv = @(v,s) (Hm+diag(s))\v;
    
    zt = quadprog(0.5*Hm,-Hm*y,[],[],[],[],low,up);
    tic
    z  = feval(mfilename,H,Hinv,y,low,up,'c',0,'out',2);
    toc
    
    errz = norm(z-zt)/norm(zt); 
    fprintf('Error when compared to Matlab quadprog: %3.2e\n',errz)
    return
end

if isempty(find(y<low | y>up,1)) % if y satisfies the bound constraints, return
   z = y;
   resid = [0,0];
   return
end

sigma   = 0.01;   % central path parameter
tau     = 0.99;   % step size selection
c       = 1e-3;   % Hessian shift
tol     = 1e-12;  % tolerance for ||Rprime||_2 and ||Rdual||_2
maxIter = 200;    % same stopping crit as quadprog
z       = [];
out     = 1;      % (0 -> no printing, 1 -> print final status, 2 -> print each iteration)
w = [];
lambda = ones(2*numel(y),1);
resid = zeros(maxIter,2);
for k=1:2:length(varargin) % overwrite default parameter
    eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

n  = numel(y);
Hy = H(y,c*ones(n,1));
b  = [low;-up];

if isempty(z)
    z = (up+low)/2;
end
if isempty(w)
    w = [z;-z]-b+1e-2;
end

%------------Initialization------------------------------------------------
iter = 0;

if out >= 1
    fprintf('=== %s (maxIter: %d, tol=%d) ===\n',mfilename,maxIter,tol);
    fprintf('iter\t Prime Residue \tDual Residue \n')
end

%------------Refine initial guess------------------------------------------
% For details, see Alg 16.4 in Nocedal/Wright
rd = H(z,c*ones(n,1)) - Hy - lambda(1:n, 1) + lambda(n+1:2*n, 1);
rp = [z;-z] - b - w;
rl = w.*lambda;

[~, delta_w_aff, delta_lambda_aff] = solveKKT(Hinv, lambda, w, c, rd, rp, rl);

w = max(1, abs(w+delta_w_aff));
lambda = max(1, abs(lambda+delta_lambda_aff));

rd = H(z,c*ones(n,1)) - Hy - lambda(1:n, 1) + lambda(n+1:2*n, 1);
rp = [z;-z] - b - w;
mu = dot(w,lambda) / (2*n); % duality measure
rl = w.*lambda-sigma*mu; 

%--------Interior point iterative loop-------------------------------------
while iter <= maxIter
    
    [dz, dw, dlam] = solveKKT(Hinv, lambda, w, c, rd, rp, rl);
    %---------------Determine step size----------------------------------------
    w_idx = find(dw<0);
    alpha_prime = min((-tau* w(w_idx)) ./ dw(w_idx)); % determine largest possible step size for prime variables
    lambda_idx = find(dlam<0);
    alpha_dual = min((-tau* lambda(lambda_idx)) ./ dlam(lambda_idx)); % determine largest possible step size for dual variables
    
    if size(lambda_idx,1) ~= 0
        if size(w_idx,1) ~= 0
            alpha = min(alpha_prime,alpha_dual); % take minimum to ensure feasibility of w and lambda
        else
            alpha = alpha_dual;
        end
    else
        if size(w_idx,1) ~= 0
            alpha = alpha_prime;
        else
            alpha = 1;
        end
    end

    %---------------Update variables-------------------------------------------
    z = z + alpha * dz;
    w = w + alpha * dw;
    lambda = lambda + alpha * dlam;
        
    rd = H(z,c*ones(n,1)) - Hy - lambda(1:n, 1) + lambda(n+1:2*n, 1);
    rp = [z;-z] - b - w;
    rl = w.*lambda-sigma*mu;
    mu = dot(w,lambda) / (2*n); % duality measure
    
    %---------------Update error and no. of iteration--------------------------
    iter = iter + 1; % count the number of iteration
    resid(iter,:) = [norm(rp),norm(rd)];
    
    if out==2; fprintf('%3d.0\t%3.2e\t%3.2e\n',iter,resid(iter,1),resid(iter,2)); end

    if resid(iter,1) < tol && resid(iter,2) < tol
        if out == 2
            fprintf('Iteration stops brcause ||Rprime||_2<tol and ||Rdual||_2<tol \n')
        end
        break
    end
end

if out==1; fprintf('%3d.0\t%3.2e\t%3.2e\n',iter,resid(iter,1),resid(iter,2)); end

if iter < maxIter
   resid(iter+1:end,:) = []; 
end

function [dz, dw, dlam] = solveKKT(Hinv, lambda, w, c, rd, rp, rl)

% Solve the system:

%	      |    H+cI       0   -\tilde{I}^T| |Δz|     |   -rd  |
%	      | \tilde{I}    -I         0     | |Δw|  =  |   -rp  |
%	      |     0      diag(λ)    diag(w) | |Δλ|     |   -rl  |

% where H = VRT'*VRT = VTV', \tilde{I} = [I -I]^T

%-----------Compute the R.H.S. of the normal equation----------------------
w      = reshape(w,[],2);
rl     = reshape(rl,[],2);
rp     = reshape(rp,[],2);
lambda = reshape(lambda,[],2);

invwlambda = lambda./w;

times_invvlambda_temp_right = invwlambda.*(-rp-rl./lambda);
right_z = -rd + times_invvlambda_temp_right(:,1) - times_invvlambda_temp_right(:,2); % R.H.S. of the normal equation

%-----------Solve the normal equation by Woodbury matrix identity----------
invdiags = sum(invwlambda,2) + c; % store diag(s)^-1
dz = Hinv(right_z,invdiags);

dlam = times_invvlambda_temp_right - invwlambda .* [dz -dz];
dw = [dz -dz] + rp;

dz = dz(:);
dw = dw(:);
dlam = dlam(:);
