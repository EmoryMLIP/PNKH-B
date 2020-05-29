function [J,dJ,H,MRwholeinv,MLwholeinv,MRwhole,MLwhole,Err] =  classObjFun(W,Z,C,paramRegW,paramClass,varargin)

%==============================================================================
% This code is modified from a part of the course materials for
% Numerical Methods for Deep Learning
% For details and license info see https://github.com/IPAIopen/NumDL-MATLAB
%==============================================================================
%
% [J,dJ,H] =  classObjFun(W,Z,C,paramRegW,paramClass,varargin)
%
% classification objective function
%
%  J(W) = softMax(W,Z,C) + genTikhonow(W,paramRegW)
%
% Input:
%  W          - current weights
%  Z          - feature matrix
%  C          - labels
%  paramRegW  - struct, parameters for regularizer
%  paramClass - struct, parameters for classification
%
% Output:
%  J            - current value of objective fuction
%  dJ           - gradient
%  H            - approximate Hessian
%  MR and ML's  - precondtioners and their inverses, which are identity in
%                 this case
%  Err          - training and validation error

if nargout==1

J = softMax(W,Z,C);

if exist('paramRegW','var') && isstruct(paramRegW)
    [Sc,~,~] = genTikhonov(W,paramRegW);
    J = J+ Sc;
end
end


if nargout>1

[J,dJ,H] = softMax(W,Z,C);

if exist('paramRegW','var') && isstruct(paramRegW)
    [Sc,dS,d2S] = genTikhonov(W,paramRegW);
    J = J+ Sc;
    dJ = dJ + dS;
    H = @(x) H(x) + d2S(x);
end
end

if nargout > 3
   MRwholeinv = @(x) x;
   MLwholeinv = @(x) x;
   MRwhole = @(x) x;
   MLwhole = @(x) x;
end

if nargout > 7
    if exist('paramClass','var') && isstruct(paramClass)
        nc = paramClass.nc; m = paramClass.m;
        C = paramClass.C; Cv = paramClass.Cv;
        trainlastlayer = paramClass.trainlastlayer;
        testlastlayer = paramClass.testlastlayer;
        
        W = reshape(W,nc,m+1);
        Strain = W*trainlastlayer;
        S      = W*testlastlayer;
        % the probability function
        htrain = exp(Strain)./sum(exp(Strain),1);
        h      = exp(S)./sum(exp(S),1);

        % Find the largest entry at each row
        [~,ind] = max(h,[],1);
        Cvpred = zeros(size(Cv));
        Ind = sub2ind(size(Cv),ind,1:size(Cv,2));
        Cvpred(Ind) = 1;
        [~,ind] = max(htrain,[],1);
        Cpred = zeros(size(C));
        Ind = sub2ind(size(Cpred),ind,1:size(Cpred,2));
        Cpred(Ind) = 1;
        trainErr = 100*nnz(abs(C-Cpred))/2/nnz(C);
        valErr   = 100*nnz(abs(Cv-Cvpred))/2/nnz(Cv);
        Err = [trainErr, valErr];
    end
end