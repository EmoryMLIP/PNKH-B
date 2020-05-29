function [T,V] = lanczosTridiag(A,b,k,tol,doReorth)
% function [T,V] = lanczosTridiag(A,b,k,tol,doReorth)
%
%   Lanczos method for computing the factorization
%
%       A = Vk*Tk*Vk',
%
%   where A is a real symmetric n by n matrix, Tk is a tridiagonal k by k 
%   matrix and the columns of the n by k matrix Vk are orthogonal.
%
%   Implementation follows:
%       Paige, C. C. (1972). 
%       Computational variants of the Lanczos method for the eigenproblem. 
%       IMA Journal of Applied Mathematics. 
%
%   Input:
%           A - function computing A*x, e.g., x -> A*x
%           b - right hand side vector
%           k - dimension of Krylov subspace
%         tol - stopping tolerance
%    doReorth - (default=false) set to true to perform full reorthogonalization
% 
%   Output:
%           T - sparse tridiagonal matrix
%           V - basis vectors
%

n   = length(b);

if nargin <5
    doReorth = 0;
end

% pre-allocate space for tridiagonalization and basis
beta  = zeros(k,1);
alpha = zeros(k,1);
V     = zeros(n,k);


beta(1) = norm(b);
V(:,1)  = b/beta(1);
if isnumeric(A)
    u = A*V(:,1);
elseif isa(A,'function_handle') 
    u = A(V(:,1));
end
    
for j=1:k-1
    alpha(j) = V(:,j)'*u;
    u        = u - alpha(j)*V(:,j);
    if doReorth % full re-orthogonalization
        for i=1:j
            u = u - V(:,i)*(V(:,i)'*u);
        end
    end
    gamma     = norm(u);
    V(:,j+1)  = u/gamma;
    beta(j+1) = gamma;
    if beta(j+1)<tol
        break;
    end
    if isnumeric(A)
        u = A*V(:,j+1) - beta(j+1)*V(:,j);
    elseif isa(A,'function_handle') 
        u = A(V(:,j+1)) - beta(j+1)*V(:,j);
    end
end

T = diag(beta(2:j),-1) + diag(alpha(1:j),0) + diag(beta(2:j),1);
V = V(:,1:j);

end