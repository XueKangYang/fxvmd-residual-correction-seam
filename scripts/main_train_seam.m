%% ============================================================
%  Main training script: SEAM-only f-x VMD + residual correction
%
%  Public GitHub version:
%  1) uses only SEAM synthetic seismic data;
%  2) does not include or load private field datasets;
%  3) does not use previously trained parameters;
%  4) calls the separately provided functions/fxvmd.m.
%% ============================================================

clear; close all; clc;

%% ---------------- Project paths ----------------
thisFile = mfilename('fullpath');
projectRoot = fileparts(fileparts(thisFile));
addpath(genpath(fullfile(projectRoot, 'functions')));
addpath(genpath(fullfile(projectRoot, 'scripts')));

rng(2025);

%% ---------------- User configuration ----------------
cfg = struct();
cfg.segyPath = fullfile(projectRoot, 'data', 'raw', 'SEAM_Interpretation_Challenge_1_Time.sgy');

% Patch size. Use smaller values first if GPU memory is limited.
cfg.recordLength = 700;   % time samples
cfg.numTraces    = 400;   % traces / inlines
cfg.numSamples   = 120;   % number of generated SEAM training samples

% AWGN range for generating noisy SEAM sections.
cfg.snrRange = [0, 10];
cfg.normalizePctl = 99;

% Optional crop ranges. Leave empty to sample from the full valid range.
cfg.timeStartRange  = [];
cfg.traceStartRange = [];

% Generated dataset cache. Delete this file if you want to regenerate samples.
cfg.outputFile = fullfile(projectRoot, 'data', 'processed', 'seam_train_700x400.mat');

% f-x VMD parameters. fxvmd.m is the separately provided version.
flow = 1;
fhigh = 150;
numIMFs = 8;
removeN = 1;
verb = 0;

% Network parameters.
depth = 2;
baseNumFilters = 32;
maxEpochs = 60;
initialLearnRate = 1e-4;
miniBatchSize = 2;
executionEnvironment = 'auto';  % 'auto', 'gpu', or 'cpu'

% Output paths.
modelOutFile = fullfile(projectRoot, 'models', 'trained_unet_fxvmd_2ch_seam.mat');
metricsOutFile = fullfile(projectRoot, 'outputs', 'metrics', 'seam_test_metrics.csv');
figOutDir = fullfile(projectRoot, 'outputs', 'figures');
figDPI = 600;

%% ---------------- 1. Generate or load SEAM synthetic dataset ----------------
if exist(cfg.outputFile, 'file') == 2
    fprintf('[1/7] Loading generated SEAM dataset: %s\n', cfg.outputFile);
    S = load(cfg.outputFile, 'NoisyData', 'CleanData', 'ResidualData', 'info'); %#ok<NASGU>
    NoisyData = S.NoisyData;
    CleanData = S.CleanData;
    info = S.info;
else
    fprintf('[1/7] Generating SEAM dataset from SEGY...\n');
    [NoisyData, CleanData, ~, info] = generate_seam_dataset(cfg);
end

dt = info.dt_s;
T = cfg.recordLength;
W = cfg.numTraces;
Ntotal = size(NoisyData, 2);

fprintf('Dataset: T=%d, W=%d, N=%d, dt=%.6f s\n', T, W, Ntotal, dt);

%% ---------------- 2. Train/validation/test split ----------------
idx = randperm(Ntotal);
nTrain = max(1, floor(0.80 * Ntotal));
nVal   = max(1, floor(0.10 * Ntotal));
nTest  = Ntotal - nTrain - nVal;

if nTest < 1
    nTest = 1;
    nTrain = Ntotal - nVal - nTest;
end

idxTrain = idx(1:nTrain);
idxVal   = idx(nTrain+1:nTrain+nVal);
idxTest  = idx(nTrain+nVal+1:end);

trainNoisy = NoisyData(:, idxTrain);
trainClean = CleanData(:, idxTrain);
valNoisy   = NoisyData(:, idxVal);
valClean   = CleanData(:, idxVal);
testNoisy  = NoisyData(:, idxTest);
testClean  = CleanData(:, idxTest);

fprintf('[2/7] Split: train=%d, val=%d, test=%d\n', numel(idxTrain), numel(idxVal), numel(idxTest));

trainNoisy3D = reshape(trainNoisy, T, W, []);
trainClean3D = reshape(trainClean, T, W, []);
valNoisy3D   = reshape(valNoisy,   T, W, []);
valClean3D   = reshape(valClean,   T, W, []);
testNoisy3D  = reshape(testNoisy,  T, W, []);
testClean3D  = reshape(testClean,  T, W, []);

%% ---------------- 3. f-x VMD preprocessing ----------------
fprintf('[3/7] Running f-x VMD preprocessing using functions/fxvmd.m...\n');

XTrain_FXVMD = zeros(T, W, size(trainNoisy3D,3));
for i = 1:size(trainNoisy3D,3)
    XTrain_FXVMD(:,:,i) = fxvmd(trainNoisy3D(:,:,i), flow, fhigh, dt, removeN, verb, numIMFs);
    if mod(i, 10) == 0 || i == 1
        fprintf('  Train f-x VMD: %d/%d\n', i, size(trainNoisy3D,3));
    end
end

XVal_FXVMD = zeros(T, W, size(valNoisy3D,3));
for i = 1:size(valNoisy3D,3)
    XVal_FXVMD(:,:,i) = fxvmd(valNoisy3D(:,:,i), flow, fhigh, dt, removeN, verb, numIMFs);
end

XTest_FXVMD = zeros(T, W, size(testNoisy3D,3));
for i = 1:size(testNoisy3D,3)
    XTest_FXVMD(:,:,i) = fxvmd(testNoisy3D(:,:,i), flow, fhigh, dt, removeN, verb, numIMFs);
end

%% ---------------- 4. Construct two-channel inputs ----------------
fprintf('[4/7] Constructing two-channel input [R_fx, F] and residual target...\n');

[XTrain, YTrain] = make_two_channel_pairs(trainNoisy3D, trainClean3D, XTrain_FXVMD);
[XVal,   YVal]   = make_two_channel_pairs(valNoisy3D,   valClean3D,   XVal_FXVMD);
[XTest,  ~, normTest] = make_two_channel_pairs(testNoisy3D, testClean3D, XTest_FXVMD);

%% ---------------- 5. Train network from scratch ----------------
fprintf('[5/7] Building and training network from scratch...\n');

inputSize = [T, W, 2];
lgraph = build_unet_dncnn(depth, baseNumFilters, inputSize);

options = trainingOptions('adam', ...
    'InitialLearnRate', initialLearnRate, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 50, ...
    'GradientThresholdMethod', 'l2norm', ...
    'GradientThreshold', 1, ...
    'L2Regularization', 1e-4, ...
    'ExecutionEnvironment', executionEnvironment, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

net = trainNetwork(XTrain, YTrain, lgraph, options);

if ~exist(fileparts(modelOutFile), 'dir')
    mkdir(fileparts(modelOutFile));
end
save(modelOutFile, 'net', 'cfg', 'flow', 'fhigh', 'numIMFs', 'removeN', 'depth', 'baseNumFilters', '-v7.3');
fprintf('Saved trained model to %s\n', modelOutFile);

%% ---------------- 6. Predict and evaluate held-out SEAM test sections ----------------
fprintf('[6/7] Predicting held-out SEAM test sections...\n');

predRNorm = predict(net, XTest);
if ndims(predRNorm) == 3
    predRNorm = reshape(predRNorm, T, W, 1, []);
end

Ntest = size(testNoisy3D, 3);
finalDenoised = zeros(T, W, Ntest);
fxvmdDenoised = XTest_FXVMD;

snrIn = zeros(Ntest, 1);
snrFXVMD = zeros(Ntest, 1);
snrFinal = zeros(Ntest, 1);

for k = 1:Ntest
    Rpred = predRNorm(:,:,1,k) * normTest.sigma(k) + normTest.mu(k);
    finalDenoised(:,:,k) = testNoisy3D(:,:,k) - Rpred;

    snrIn(k) = compute_snr_db(testClean3D(:,:,k), testNoisy3D(:,:,k));
    snrFXVMD(k) = compute_snr_db(testClean3D(:,:,k), fxvmdDenoised(:,:,k));
    snrFinal(k) = compute_snr_db(testClean3D(:,:,k), finalDenoised(:,:,k));
end

Metrics = table((1:Ntest)', snrIn, snrFXVMD, snrFinal, snrFinal - snrIn, ...
    'VariableNames', {'SampleID', 'SNR_Input_dB', 'SNR_FXVMD_dB', 'SNR_Final_dB', 'DeltaSNR_Final_dB'});

if ~exist(fileparts(metricsOutFile), 'dir')
    mkdir(fileparts(metricsOutFile));
end
writetable(Metrics, metricsOutFile);
fprintf('Saved metrics to %s\n', metricsOutFile);

fprintf('Mean SNR input = %.2f dB | f-x VMD = %.2f dB | final = %.2f dB\n', ...
    mean(snrIn), mean(snrFXVMD), mean(snrFinal));

%% ---------------- 7. Save example figures ----------------
fprintf('[7/7] Saving example figures...\n');

if ~exist(figOutDir, 'dir')
    mkdir(figOutDir);
end

exampleIdx = 1;
xAxis = 1:W;
tAxis = (0:T-1) * dt;

noisyExample = testNoisy3D(:,:,exampleIdx);
cleanExample = testClean3D(:,:,exampleIdx);
fxvmdExample = fxvmdDenoised(:,:,exampleIdx);
finalExample = finalDenoised(:,:,exampleIdx);

resFXVMD = noisyExample - fxvmdExample;
resFinal = noisyExample - finalExample;

ampLim = prctile(abs(noisyExample(:)), 99);
resLim = prctile(abs(resFinal(:)), 99);

plot_seismic_section(cleanExample, xAxis, tAxis, [-ampLim ampLim], ...
    'Clean SEAM section', fullfile(figOutDir, 'example_clean_seam_section.png'), figDPI);
plot_seismic_section(noisyExample, xAxis, tAxis, [-ampLim ampLim], ...
    'Noisy SEAM section', fullfile(figOutDir, 'example_noisy_seam_section.png'), figDPI);
plot_seismic_section(fxvmdExample, xAxis, tAxis, [-ampLim ampLim], ...
    'f-x VMD denoised section', fullfile(figOutDir, 'example_fxvmd_denoised_section.png'), figDPI);
plot_seismic_section(resFXVMD, xAxis, tAxis, [-resLim resLim], ...
    'f-x VMD residual section', fullfile(figOutDir, 'example_fxvmd_residual_section.png'), figDPI);
plot_seismic_section(finalExample, xAxis, tAxis, [-ampLim ampLim], ...
    'Final denoised section', fullfile(figOutDir, 'example_final_denoised_section.png'), figDPI);
plot_seismic_section(resFinal, xAxis, tAxis, [-resLim resLim], ...
    'Final residual section', fullfile(figOutDir, 'example_final_residual_section.png'), figDPI);

fprintf('Done.\n');
