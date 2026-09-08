%% Quick repository sanity demo without SEGY data
% This demo only checks path setup, helper functions, and network construction.
% It does not reproduce the full SEAM training workflow.

clear; close all; clc;

thisFile = mfilename('fullpath');
projectRoot = fileparts(fileparts(thisFile));
addpath(genpath(fullfile(projectRoot, 'functions')));

T = 128;
W = 64;
N = 4;

t = (0:T-1)'/T;
x = linspace(0, 1, W);
clean3D = zeros(T, W, N);
noisy3D = zeros(T, W, N);
fxvmd3D = zeros(T, W, N);

for k = 1:N
    section = sin(2*pi*(8*t + 0.7*x)) + 0.5*sin(2*pi*(14*t - 0.4*x));
    clean3D(:,:,k) = normalize_seismic_patch(section, 99);
    noisy3D(:,:,k) = add_awgn_measured(clean3D(:,:,k), 5);
    % For a quick demo without running VMD, use a simple moving average placeholder.
    % The full workflow in scripts/main_train_seam.m uses functions/fxvmd.m.
    fxvmd3D(:,:,k) = movmean(noisy3D(:,:,k), 5, 1);
end

[X, Y] = make_two_channel_pairs(noisy3D, clean3D, fxvmd3D);
lgraph = build_unet_dncnn(1, 8, [T, W, 2]);

fprintf('Quick demo finished. X size = [%s], Y size = [%s].\n', ...
    num2str(size(X)), num2str(size(Y)));

disp(lgraph.Layers);
