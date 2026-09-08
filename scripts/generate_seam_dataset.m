function [NoisyData, CleanData, ResidualData, info] = generate_seam_dataset(cfg)
%GENERATE_SEAM_DATASET Generate clean/noisy/residual training samples from SEAM SEGY.
%
% Required cfg fields:
%   cfg.segyPath
%   cfg.recordLength
%   cfg.numTraces
%   cfg.numSamples
%
% Optional cfg fields:
%   cfg.snrRange          default [0, 10]
%   cfg.normalizePctl     default 99
%   cfg.outputFile        default ''
%   cfg.timeStartRange    default full valid range
%   cfg.traceStartRange   default full valid range
%
% Outputs:
%   NoisyData    [T*W, N]
%   CleanData    [T*W, N]
%   ResidualData [T*W, N], noisy - clean

    arguments
        cfg struct
    end

    segyPath = cfg.segyPath;
    T = cfg.recordLength;
    W = cfg.numTraces;
    N = cfg.numSamples;

    if ~isfield(cfg, 'snrRange') || isempty(cfg.snrRange)
        cfg.snrRange = [0, 10];
    end
    if ~isfield(cfg, 'normalizePctl') || isempty(cfg.normalizePctl)
        cfg.normalizePctl = 99;
    end
    if ~isfield(cfg, 'outputFile')
        cfg.outputFile = '';
    end

    assert(exist(segyPath, 'file') == 2, 'SEAM SEGY file not found: %s', segyPath);

    segyInfo = getSEGYInfo_noToolbox(segyPath);
    fprintf('SEGY info: ns=%d, ntr=%d, dt=%.3f ms, format=%d\n', ...
        segyInfo.ns, segyInfo.ntr, segyInfo.dt_ms, segyInfo.formatCode);

    maxTimeStart = segyInfo.ns - T + 1;
    maxTraceStart = segyInfo.ntr - W + 1;
    assert(maxTimeStart >= 1 && maxTraceStart >= 1, ...
        'Patch size [%d x %d] exceeds SEGY size [%d x %d].', T, W, segyInfo.ns, segyInfo.ntr);

    if isfield(cfg, 'timeStartRange') && ~isempty(cfg.timeStartRange)
        timeStartRange = cfg.timeStartRange;
        timeStartRange(1) = max(timeStartRange(1), 1);
        timeStartRange(2) = min(timeStartRange(2), maxTimeStart);
    else
        timeStartRange = [1, maxTimeStart];
    end

    if isfield(cfg, 'traceStartRange') && ~isempty(cfg.traceStartRange)
        traceStartRange = cfg.traceStartRange;
        traceStartRange(1) = max(traceStartRange(1), 1);
        traceStartRange(2) = min(traceStartRange(2), maxTraceStart);
    else
        traceStartRange = [1, maxTraceStart];
    end

    assert(timeStartRange(1) <= timeStartRange(2), 'Invalid timeStartRange.');
    assert(traceStartRange(1) <= traceStartRange(2), 'Invalid traceStartRange.');

    NoisyData = zeros(T*W, N);
    CleanData = zeros(T*W, N);
    ResidualData = zeros(T*W, N);

    sampleInfo = struct('timeStart', cell(N,1), 'traceStart', cell(N,1), 'snrTarget', cell(N,1), 'snrActual', cell(N,1));

    for k = 1:N
        timeStart = randi(timeStartRange);
        traceStart = randi(traceStartRange);

        [patch, dt_ms] = readSEGY_patch_noToolbox(segyPath, T, W, [timeStart, traceStart]);
        clean = normalize_seismic_patch(patch, cfg.normalizePctl);

        snrTarget = cfg.snrRange(1) + rand() * (cfg.snrRange(2) - cfg.snrRange(1));
        noisy = add_awgn_measured(clean, snrTarget);
        residual = noisy - clean;

        CleanData(:, k) = clean(:);
        NoisyData(:, k) = noisy(:);
        ResidualData(:, k) = residual(:);

        sigp = mean(clean(:).^2);
        noip = mean(residual(:).^2);
        snrActual = 10 * log10(sigp / max(noip, eps));

        sampleInfo(k).timeStart = timeStart;
        sampleInfo(k).traceStart = traceStart;
        sampleInfo(k).snrTarget = snrTarget;
        sampleInfo(k).snrActual = snrActual;

        if mod(k, max(1, floor(N/10))) == 0 || k == 1
            fprintf('Generated sample %d/%d | timeStart=%d traceStart=%d SNR=%.2f dB\n', ...
                k, N, timeStart, traceStart, snrActual);
        end
    end

    info = struct();
    info.T = T;
    info.W = W;
    info.N = N;
    info.dt_ms = dt_ms;
    info.dt_s = dt_ms / 1000;
    info.segyInfo = segyInfo;
    info.sampleInfo = sampleInfo;

    if ~isempty(cfg.outputFile)
        outDir = fileparts(cfg.outputFile);
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end
        save(cfg.outputFile, 'NoisyData', 'CleanData', 'ResidualData', 'info', '-v7.3');
        fprintf('Saved generated dataset to %s\n', cfg.outputFile);
    end
end
