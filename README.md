# PNKH-B: A Projected Newton-Krylov Method for Large-Scale Bound-Constrained Optimization Problems

Kelvin Kan, Samy Wu Fung and Lars Ruthotto

## Dependencies

The MNIST example requires the current version of [NumDL-MATLAB](https://github.com/IPAIopen/NumDL-MATLAB) 

## How to run the MNIST example

The MNIST example is the numerical experiment reported in Section 5.3 of our paper

- Download [NumDL-MATLAB](https://github.com/IPAIopen/NumDL-MATLAB) 
- replace NumDL-MATLAB\classification\classObjFun.m by PNKH-B\tests\classObjFun.m 
- replace NumDL-MATLAB-master\test\Rosenbrock.m by PNKH-B\tests\Rosenbrock.m
- run PNKH-B\tests\driver_MNISTexample.m

## Acknowledgements

This material is in part based upon work supported by the National Science Foundation under Grant Number 1751636. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
