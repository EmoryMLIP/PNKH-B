function [xc,his,xAll] = PNKH(fun,xc,varargin)

% [xc,his,xAll] = PNKH(fun,xc,varargin)
% 
% Lanzcos-based projected Newton method for bound constrained problem
%
% min_x f(x) s.t. low <= x <= up
%
% Required Inputs:
%
%  fun  - objective funciton which takes one argument, the current variable,
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
% qpIter         - Quadratic programm MaxIter
% maxIterCG      - numbers of CG iteration as a vector
% out            - print option, 0-> don't print, 1-> print each iteration  
% alpha          - line search parameter
% epsilon        - width of the boundary
% maxStep        - max step size
% factor_c       - factor for Hessian shift in the orthogonal complement
% (if specified, we will shift H by factor_c*smallest eigenvalue of T)
% (if not specified, we will shift H by the smallest eigenvalue of T)

% low            - lower bound, put a large negative number if no bound
% up             - upper bound, put a large positive number if no bound
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
   [xc,his,xAll] = feval(mfilename,E,W,...
                    'low',1.3*ones(2,1),'up',2.2*ones(2,1),'cgTol',1e-16,...
                    'indexing', 'Augmented');
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

xTol           = 1e-6;                    %Relative change tolerance
gTol           = 1e-8;                    %Projected gradient tolerance
cgTol          = 1e-16;                   %CG tolerance
qpTol          = 1e-12;                   %Quadratic programming tolerance
qpIter         = 200;                     %Quadratic programm MaxIter
maxIterCG      = ones(1,10)*9;            %number of CG iterations
out            = 1;                       %print option
alpha          = 0.1;                     %line search parameter
epsilon        = 1e-6;                    %width of the boundary
maxStep        = inf;                     %max step size
low            = -1e16*ones(size_prob,1); %lower bound
up             = 1e16*ones(size_prob,1);  %upper bound
indexing       = 'Augmented';               %indexing method
quadProgSolver = 'interiorQP';            %QP solver
for k=1:2:length(varargin) % overwrites default parameter
  eval([varargin{k},'=varargin{',int2str(k+1),'};']);
end

xAll = [];
[fc,df,H,MRwholeinv,MLwholeinv,MRwhole,MLwhole,Err] = fun(xc);
maxIter = numel(maxIterCG);
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
    fprintf('=== %s (maxIter: %d, maxIterCG(1)=%d, cgTol=%1.2e, qpTol=%1.2e) ===\n',mfilename,maxIter,maxIterCG(1),cgTol,qpTol);
    fprintf(his.str);
end

for j=1:maxIter
    r = maxIterCG(j);
    maxCG = r + 1; %Lanczos counts one less iteration
                           %when compared to CG
    s = zeros(size_prob, 1);
    switch indexing
        case 'Boundary'
            Jk = (xc <= low + epsilon) | (xc >= up - epsilon);
            Fk = not(Jk); 
        case 'Augmented'
            Jk = ((xc <= low + epsilon) & (df > 0)) | ...  
                     ((xc>= up - epsilon) & (df < 0));
            Fk = not(Jk);                                         
        otherwise                                                 % no indexing
            Jk = zeros(size_prob, 1);
            Fk = ones(size_prob, 1);
    end
    Jk = find(Jk==1); Fk = find(Fk==1);
    
    if norm(df(Fk)) && ~isempty(Fk)
    
        MRinv = @(v) MRinv_temp(v, size_prob, MRwholeinv, Fk);
        MLinv = @(v) MLinv_temp(v, size_prob, MLwholeinv, Fk);

        MR    = @(v) MR_temp(v, size_prob, MRwhole, Fk);
        ML    = @(v) ML_temp(v, size_prob, MLwhole, Fk);

        PCH = @(v) rest_Hess(v,H,Fk,size_prob,MRinv,MLinv);

        if nargout>2; xAll = [xAll xc]; end;
        [Activedf, Activeidx] = projectGradient(df,xc,low,up);
        

        [T,V] = lanczosTridiag(PCH,MLinv(df(Fk)),maxCG,cgTol,1);

        s(Fk) = MRinv(V*(T\(V'*MLinv(-df(Fk)))));
        [V, QR_R] = qr(ML(V), 0);
        T = QR_R*T*QR_R';
        [eig_V, eig_D] = eig(T);
        V = V*eig_V;
        
        [c, min_idx] = min(diag(eig_D));
        
        if exist('factor_c', 'var')
            c = c*factor_c;
        else
            V(:,min_idx)=[];
            eig_D(:,min_idx)=[]; eig_D(min_idx,:)=[];
        end

        T = eig_D-c*eye(size(eig_D,1));
        
        R = chol(T); VRT = V*R'; % define this here so that we dont have to do it in each IPM
        LowRankH    = @(v,s) V*(T*(V'*v))+s.*v;
        Identity_map = @(x) x;
        LowRankHinv = @(v,s) LowRankHinv_temp(v, s, V, T, Identity_map, Identity_map, VRT);

        CGrelerr = norm(PCH(MR(s(Fk)))+MLinv(df(Fk)))/norm(MLinv(df(Fk))); 
        his.obj(j,1:end-2) = [fc,norm(Activedf),nnz(Activeidx)/numel(xc),CGrelerr,Err];
        % Compute Lanczos relative error (consistent with Matlab pcg)

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
        switch indexing
            case 'Boundary' %Rescale gradient based on Newton step
                ga = -df(xc <= low + epsilon);
                if max(abs(ga)) > max(abs(s(Fk))), ga = ga/max(abs(ga))*max(abs(s(Fk))); end
                s(xc <= low + epsilon) = ga;
                ga = -df(xc >= up - epsilon);
                if max(abs(ga)) > max(abs(s(Fk))), ga = ga/max(abs(ga))*max(abs(s(Fk))); end
                s(xc >= up - epsilon) = ga;
            case 'Augmented' %Rescale gradient based on epsilon of augmented index
                ga = s_Jk(s_Jk<0);
                if max(abs(ga)) > epsilon, ga = ga/max(abs(ga))*epsilon; end
                s_Jk(s_Jk<0) = ga;
                ga = s_Jk(s_Jk>0);
                if max(abs(ga)) > epsilon, ga = ga/max(abs(ga))*epsilon; end
                s_Jk(s_Jk>0) = ga;    
        end
        s(Jk) = s_Jk;

    else
        %if norm(dk(Fk))=0 or Fk is empty, 
        %then we only use gradient descent
        s = -df/norm(df);
    end
    
    % Armijo line search
    cnt = 1;
    while 1
        xtry = xc(Fk) + mu*s(Fk);
        % projection
        % QP solver
    if norm(df(Fk)) && ~isempty(Fk)
        %if norm(dk(Fk))=0 or Fk is empty, 
        %then we only use gradient descent
        %and we skip QP 
        if strcmp(quadProgSolver, 'matlabQuadProg') 
            options =  optimset('Display','off');
            Hv = V*T*V'+c*speye(size(V,1));
            Hv = (Hv+Hv')/2;
            xtry = quadprog(Hv,-Hv*xtry,[],[],[],[],low(Fk),up(Fk),[],options);
        elseif strcmp(quadProgSolver, 'interiorQP')
            [xtry, error_xtry] = interiorQPB(LowRankH,LowRankHinv,xtry,low(Fk),up(Fk),...
                                            'out',0, 'tol', qpTol, 'c', c, 'maxIter', qpIter);
        end
    end
        temp_xtry = xtry;
        xtry = zeros(size_prob,1);
        xtry(Fk) = temp_xtry;
        
        temp_xtry2 =  xc(Jk) + mu * s(Jk);
        xtry(Jk) = min(max(temp_xtry2,low(Jk)),up(Jk));
        
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
    [fc,df,H,MRwholeinv,MLwholeinv,MRwhole,MLwhole,Err] = fun(xc);
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
   % v         - vector with length equals to Fk
   % H         - Original Hessian (function handle)
   % Fk        - index of inactive sets
   % size_prob - size of the problem
   % MRinv     - right preconditioner s.t. MLinv*H*MRinv*(MR*x) = MLinv*(-\nabla f)
   % MLinv     - left preconditioner  s.t. MLinv*H*MRinv*(MR*x) = MLinv*(-\nabla f)
   % 
   % Output:
   % x         - H|_{Fk} * v|_{Fk}, where |_{Fk} means restricted onto Fk
   temp = zeros(size_prob,1);
   temp(Fk) = MRinv(v);
   x_with_zero = H(temp);
   x = MLinv(x_with_zero(Fk));
end

% function x = rest_Hess_nopre(v,H,Fk,size_prob) %restricted Hessian, input: restricted vector
%    temp = zeros(size_prob,1);
%    temp(Fk) = (v);
%    x_with_zero = H(temp);
%    x = x_with_zero(Fk);
% end

function x = MR_temp(v, size_prob, MRwhole, Fk)
   % restricted right preconditioner (function handle)
   % Required inputs:
   % v            - vector with length equals to Fk
   % size_prob    - size of the problem
   % MRwhole      - Orginal right precondtioner
   % Fk           - index of inactive sets
   % 
   % Output:
   % x            - MRwhole|_{Fk} * v|_{Fk}, where |_{Fk} means restricted onto Fk
    x = zeros(size_prob,size(v,2));
    x(Fk,:) = v;
    x = MRwhole(x);
    x = x(Fk,:);
end

function x = ML_temp(v, size_prob, MLwhole, Fk)
   % restricted left preconditioner (function handle)
   % Required inputs:
   % v            - vector with length equals to Fk
   % size_prob    - size of the problem
   % MLwhole      - Orginal left precondtioner
   % Fk           - index of inactive sets
   % 
   % Output:
   % x            - MLwhole|_{Fk} * v|_{Fk}, where |_{Fk} means restricted onto Fk
    x = zeros(size_prob,size(v,2));
    x(Fk,:) = v;
    x = MLwhole(x);
    x = x(Fk,:);
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

function x = LowRankHinv_temp(v, s, V, T, MRinv, MLinv, VRT)
    % The inverse of the low-rank approximated Hessian (function handle)
    % Required inputs:
    % v            - vector with length equals to Fk
    % s            - vector that shifts the Hessian to ensure symmetric
    %                positive definiteness
    % V, T         - output of Lanczos tridiagonalization
    % MRinv, MLinv - inverse preconditioner restricted onto Fk
    % VRT          - ML*V*R', where T=R'*R, cholesky factorization of T
    %
    % Output:
    % x            - (V*T*V'+diag(s))^(-1)*v
    if norm(s) == 0
        x = MRinv(V*(T\(V'*MLinv(v))));
    else
        opts.SYM = true; opts.POSDEF = true;
        sinv = s.^(-1); low_rank = size(VRT,2);
        x = linsolve(VRT'*(sinv.*VRT)+eye(low_rank),VRT'*(sinv.*v),opts);
        x = sinv .* v - sinv .* (VRT*x); 
    end
end
