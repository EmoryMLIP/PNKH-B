function [df, idx] = projectGradient(df,xc,low,up,epsilon)

% Project the gradient onto the inactive set, that is,
% keep the entries in inactive set = (((xc <= low+epsilon) & (df > 0)) | ((xc>= up-epsilon) & (df < 0)))
% and make the other entries zeros.
% Also return the index of the inactive set.

% Required Inputs:
% df      -  gradient
% xc      -  variable
% low     -  vector of lower bounds
% up      -  vector of upper bounds
% epsilon -  Width of the boundary
%
% Outputs:
% df      -  Projected gradient
% idx     -  index of the inactive variables

if not(exist('epsilon','var')) || isempty(epsilon)
    epsilon = 1e-6;
end

idx = (xc<=low+epsilon & df>0) | (xc>=up-epsilon & df<0);
df(idx) = 0;