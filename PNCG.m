function [xc,his,xAll] = PNCG(fun,xc,varargin)

% [xc,his,xAll] = PNCG(fun,xc,varargin)
% 
% Projected Newton-CG for bound constrained problem
%
% min_x f(x) s.t. low <= x <= up
%
% Required Inputs:
%
%  fun  - objective funciton which takes one argument, the current variable
%         and it outputs [f, df, H, MRwholeinv, MLwholeinv, MRwhole, ...
%         MLwhole, Err],
%         where f is objective function value, df is gradient, H is Hessian
%         MRwholeinv, MLwholeinv, MRwhole, MLwhole are preconditioners and
%         their inverses, 
%         Err is a row vector containing the training and validation error
%  xc   - initial guess
%
% Optional Inputs (configured via varargin, see code for defaults)
%
% xTol           - Relative change (||x_{k+1}-x_k||_2/||x_k||_2) tolerance
% gTol           - Projected gradient tolerance
% cgTol          - CG tolerance
% qpTol          - Quadratic programming tolerance
% maxIterCG      - number of CG iteration
% out            - print option, 0-> don't print, 1-> print each iteration  
% alpha          - line search parameter
% epsilon        - width of the boundary
% maxIter        - max number of iterations
% maxStep        - max step size
% low            - lower bound
% up             - upper bound
% indexing       - indexing method
% quadProgSolver - QP solver
%
% Outputs
%
% xc   - optimal variable
% his  - structure containing the history of iterations
%        his.str: corresponding value in each column
%        his.obj: values of history of iterations,
%        each row is one iteration and is
%        [objtive function value, norm of restricted gradient, ...
%        percentage of active variables, CG relative error, ...
%        training error, validation error, mu, number of function evaluations];
% xAll - collection of all x's at all iterations

if nargin==0
   E = @(x,varargin) Rosenbrock(x);
   W = [2;2];
   [xc,his,xAll] = feval(mfilename,E,W,'maxIter',40,...
                    'low',1.3*ones(2,1),'up',2.2*ones(2,1),'cgTol',1e-16,...
                    'indexing', 'Epsilon', 'xTol', 1e-10);
   fprintf('numerical solution: x = [%1.4f, %1.4f]\n',xc);
   figure(1); clf;
   subplot(1,2,1)
   x1= linspace(1.3,2.2,128);
   x2= linspace(1.3,2.2,128);
   [X1,X2] = meshgrid(x1,x2);
   F = reshape(Rosenbrock([X1(:) X2(:)]'),128,128);
   [fmin,idmin] = min(F(:));
   xmin = [X1(idmin); X2(idmin)];
   contour(X1,X2,F,100,'LineWidth',3)
   hold on;
   plot(xmin(1),xmin(2),'sb','MarkerSize',50);
   plot(xc(1),xc(2),'.r','MarkerSize',30);
   subplot(1,2,2)
   semilogy(his.obj(:,2),'LineWidth',3);
   set(gca,'FontSize',20);
   title('optimality condition');
   return
end
size_prob = size(xc,1);
warning('off', 'MATLAB:pcg:tooSmallTolerance')

xTol           = 1e-6;                    %Relative change tolerance
gTol           = 1e-8;                    %Projected gradient tolerance
cgTol          = 1e-16;                   %CG tolerance
maxIterCG      = 10;                      %number of CG iterations
out            = 1;                       %print option
alpha          = 0.1;                     %line search parameter
epsilon        = 1e-6;                    %width of the boundary
maxIter        = 20;                      %max number of iterations
maxStep        = inf;                     %max step size
low            = -1e16*ones(size_prob,1); %lower bound
up             = 1e16*ones(size_prob,1);  %upper bound
indexing       = 'Epsilon';               %indexing method

for k=1:2:length(varargin)     % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

xAll = [];
[fc,df,H,MRwholeinv,MLwholeinv,~,~,Err] = fun(xc);
mu = 1; 

if isempty(Err)
    his.str = 'iter\t obj func\tnorm(grad)\tActive \t CGrelerr \n';
    his.val = '%3d.0\t%3.2e\t%3.2e\t%3.2e\t%3.2e\n';
else
    his.str = 'iter\t obj func\tnorm(grad)\tActive \t CGrelerr \t\tTrain Error \tVali Error \n';
    his.val = '%3d.0\t%3.2e\t%3.2e\t%3.2e\t%3.2e\t%8.2f\t%10.2f \n';
end
his.obj = zeros(maxIter+1,6+2*(~isempty(Err)));
if out==1
    fprintf('=== %s (maxIter: %d, maxIterCG=%d, cgTol=%1.2e) ===\n',mfilename,maxIter,maxIterCG,cgTol);
    fprintf(his.str);
end

for j=1:maxIter
    s = zeros(size_prob, 1);
    switch indexing
        case 'Boundary'
            Jk = (xc <= low + epsilon) | (xc >= up - epsilon);
            Fk = not(Jk); 
        case 'Epsilon'
            Jk = ((xc <= low + epsilon) & (df > 0)) | ...  
                     ((xc>= up - epsilon) & (df < 0));
            Fk = not(Jk);                                       
    end
    Jk = find(Jk==1); Fk = find(Fk==1);
    
    MRinv = @(v) MRinv_temp(v, size_prob, MRwholeinv, Fk);
    MLinv = @(v) MLinv_temp(v, size_prob, MLwholeinv, Fk);

    PCH = @(v) rest_Hess(v,H,Fk,size_prob,MRinv,MLinv);
    
    if nargout>2; xAll = [xAll xc]; end;
    
    [sTemp,FLAG,RELRES,ITER,RESVEC] = pcg(PCH,-MLinv(df(Fk)),cgTol,maxIterCG);
    
    s(Fk) = MRinv(sTemp);
    
    [Activedf, Activeidx] = projectGradient(df,xc,low,up);
    his.obj(j,1:end-2) = [fc,norm(Activedf),nnz(Activeidx)/numel(xc),RELRES,Err];
    if out==1; fprintf(his.val,j,his.obj(j,1:end-2)); end
    
    if his.obj(j,2) < gTol %if projected gradient is too small, stop
        fprintf('Iteration stops early because norm(ProjGrad)<gTol');
        fprintf('\n')
        break
    end
    if j>1 && xchange < xTol
        fprintf('Iteration stops early because ||x_{k+1}-x_k||_2/||x_k||_2<xTol')
        fprintf('\n')
        break
    end
    %------------------Rescale Inactive step-------------------------------
    if max(abs(s(Fk))) > maxStep
            fprintf('maxstep Reached...\n');
            s(Fk) = s(Fk)/max(abs(s(Fk)))*maxStep; 
    end
    %------------------Rescale Active step---------------------------------
    s_Jk = -df(Jk);
    maxsFK = max(abs(s(Fk)));
    switch indexing
        case 'Boundary' %Rescale gradient based on Newton step
            ga = -df(xc <= low);
            if max(abs(ga)) > maxsFK, ga = ga/max(abs(ga))*maxsFK; end
            s(xc <= low) = ga;
            ga = -df(xc >= up);
            if max(abs(ga)) > maxsFK, ga = ga/max(abs(ga))*maxsFK; end
            s(xc >= up) = ga;
        case 'Epsilon' %Rescale gradient based on epsilon of epsilon index
            ga = s_Jk(s_Jk<0);
            if max(abs(ga)) > maxsFK, ga = ga/max(abs(ga))*maxsFK; end
            s_Jk(s_Jk<0) = ga;
            ga = s_Jk(s_Jk>0);
            if max(abs(ga)) > maxsFK, ga = ga/max(abs(ga))*maxsFK; end
            s_Jk(s_Jk>0) = ga;    
    end
    s(Jk) = s_Jk;
    % resort to steepest descent if pcg fails
    if norm(s(Fk))==0
        s = -df/norm(df);
    end
    
    % test if s is a descent direction
    if s(:)'*df(:) > 0
        s = -df/norm(df);
    end
    
    % Armijo line search
    cnt = 1;
    while 1
        xtry = xc + mu*s;
        xtry = min(max(xtry,low),up);
        
        objtry = fun(xtry);
        if out==1; fprintf('%3d.%d\t%3.2e\t mu=%3.2e\n',j,cnt,objtry,mu); end
        if objtry < fc + alpha * dot(df, xtry-xc)
            break
        end
        mu = mu/2;
        cnt = cnt+1;
        if cnt > 10
            warning('Line search break');
            return;
        end
    end
    if cnt == 1
        mu = min(mu*1.5,1);
    end    
    his.obj(j,end-1:end) = [mu, cnt];
    xchange = norm(xtry-xc)/norm(xc);
    xc   = xtry;
    [fc,df,H,MRwholeinv,MLwholeinv,~,~,Err] = fun(xc);
end
if his.obj(maxIter,end) ~= 0 %If it does not stop early, get information of the last iter
    if nargout>2; xAll = [xAll xc]; end;
    [Activedf, Activeidx] = projectGradient(df,xc,low,up);
    his.obj(maxIter+1,1:end-2) = [fc,norm(Activedf),nnz(Activeidx)/numel(xc),0,Err];
else % If it stops early
    his.obj = his.obj(1:j,:);
end

end

function x = rest_Hess(v,H,Fk,size_prob,MRinv,MLinv) 
   % preconditioned restricted Hessian (function handle)
   % Required inputs:
   % v            - vector with length equals to Fk
   % H            - Original Hessian (function handle)
   % Fk           - index of inactive sets
   % size_prob    - size of the problem
   % MRinv, MLinv - inverse right and left preconditioner 
   %                restricted onto Fk s.t.
   %                MLinv*H|_{Fk}*MRinv*(MR*x) = MLinv*(-\nabla f|_{Fk})
   % 
   % Output:
   % x         - H|_{Fk} * v|_{Fk}, where |_{Fk} means restricted onto Fk
   temp = zeros(size_prob,1);
   temp(Fk) = MRinv(v);
   x_with_zero = H(temp);
   x = MLinv(x_with_zero(Fk));
end

function x = MRinv_temp(v, size_prob, MRwholeinv, Fk)
   % restricted inverse right preconditioner (function handle)
   % Required inputs:
   % v            - vector with length equals to Fk
   % size_prob    - size of the problem
   % MRwholeinv   - Orginal inverse right precondtioner
   % Fk           - index of inactive sets
   % 
   % Output:
   % x            - MRwholeinv|_{Fk} * v|_{Fk}, where |_{Fk} means restricted onto Fk
    x = zeros(size_prob,size(v,2));
    x(Fk,:) = v;
    x = MRwholeinv(x);
    x = x(Fk,:);
end

function x = MLinv_temp(v, size_prob, MLwholeinv, Fk)
   % restricted inverse right preconditioner (function handle)
   % Required inputs:
   % v            - vector with length equals to Fk
   % size_prob    - size of the problem
   % MLwholeinv   - Orginal inverse left precondtioner
   % Fk           - index of inactive sets
   % 
   % Output:
   % x            - MLwholeinv|_{Fk} * v|_{Fk}, where |_{Fk} means restricted onto Fk
    x = zeros(size_prob,size(v,2));    
    x(Fk,:) = v;
    x = MLwholeinv(x);
    x = x(Fk,:);
end