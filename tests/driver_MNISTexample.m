close all; clear all;
rng('default')
rng(1)
train_size = 50000; test_size = 10000;

%%
[Y,C,Yv,Cv] = setupMNIST(train_size,test_size);
C = C(1:10,:);
Cv = Cv(1:10,:);

Y = reshape(Y,[],train_size);
Yv = reshape(Yv, [], test_size);

idx = (sum(C,1)>0);
Y = Y(:,idx);
C = C(:,idx);

idv = (sum(Cv,1)>0);
Yv = Yv(:,idv);
Cv = Cv(:,idv);

ELM = true;

%% optimize
if ELM
    m  = 4000;
    nf = size(Y,1);
    nc = size(C,1);
    KOpt = randn(m,nf);
    bOpt = randn(m,1);
    Z = tanh(KOpt*Y+bOpt); % single layer
    Zv = tanh(KOpt*Yv+bOpt); % single layer
else
    %------------no extreme learning machine----------------------------------
    m  = 784;
    nf = size(Y,1);
    nc = size(C,1);
    Z  = Y; Zv = Yv; bOpt = 0; KOpt = eye(m,nf);
    %------------no extreme learning machine----------------------------------
end

%% setup parameter
W0 = randn(nc,m+1)/sqrt(nc*m);
up = 0.05*ones(numel(W0),1); low = -0.05*ones(numel(W0),1); %tried 0 and -0.05
W0(:) = max(min(W0(:), up), low);
maxIterCG = ones(1,20)*20;
cgTol     = 1e-16;
regpar    = 0;

paramClass.nc = nc; paramClass.m = m;
paramClass.C = C; paramClass.Cv = Cv;
paramClass.trainlastlayer = padarray(Z,[1,0],1,'post');
paramClass.testlastlayer  = padarray(Zv,[1,0],1,'post');
paramReg = struct('L',speye(numel(W0)),'lambda',regpar);
fctn = @(x,varargin) classObjFun(x,Z,C,paramReg,paramClass);

save_option = false;

%% --------------------------------------------------------------------------

fprintf('Performing Method 1 (PNCG_boundary)\n')
tic;
[~,his.m1] = PNCG(fctn, W0(:), 'maxIterCG', maxIterCG,...
    'cgTol', cgTol, 'low', low, 'up', up, ...
    'indexing', 'Boundary');
elapsedTime.m1 = toc;

%% --------------------------------------------------------------------------

fprintf('Performing Method 2 (PNCG_augmented)\n')
tic;
[~,his.m2] = PNCG(fctn, W0(:), 'maxIterCG', maxIterCG,...
    'cgTol', cgTol, 'low', low, 'up', up, ...
    'indexing', 'Augmented');
elapsedTime.m2 = toc;

%% --------------------------------------------------------------------------
fprintf('Performing Method 3 (PNKH-B)\n')
tic;
[~,his.m3] = PNKH(fctn, W0(:), 'maxIterCG', maxIterCG,...
    'cgTol', cgTol, 'low', low, 'up', up,...
    'indexing', 'No Index');
elapsedTime.m3 = toc;

%% --------------------------------------------------------------------------

fprintf('Performing Method 4 (PNKH-B_boundary)\n')
tic;
[~,his.m4] = PNKH(fctn, W0(:), 'maxIterCG', maxIterCG,...
    'cgTol', cgTol, 'low', low, 'up', up, ...
    'indexing', 'Boundary');
elapsedTime.m4 = toc;

%% --------------------------------------------------------------------------

fprintf('Performing Method 5 (PNKH-B_augmented)\n')
tic;
[~,his.m5] = PNKH(fctn, W0(:), 'maxIterCG', maxIterCG,...
    'cgTol', cgTol, 'low', low, 'up', up, ...
    'indexing', 'Augmented');
elapsedTime.m5 = toc;

%% --------------------------------------------------------------------------

% save results
if save_option
    FileName = strcat('MNISTresult(NewtonKrylov)', num2str(numel(maxIterCG)), '_CG', num2str(maxIterCG(end)), '_L', ...
        num2str(mean(low)), '_U',num2str(mean(up)), '_reg', num2str(regpar), '.mat');
    param = struct('maxIter', numel(maxIterCG), 'maxIterCG', maxIterCG, 'low', low, 'up', up, 'regpar', regpar);
    save(FileName, 'his', 'elapsedTime', 'param');
end
