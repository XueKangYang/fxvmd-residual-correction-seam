function [patch, dt_ms, meta] = readSEGY_patch_noToolbox(segyPath, T, W, fixedPos)
%READSEGY_PATCH_NOTOOLBOX Read a 2-D SEGY patch without the MATLAB SEGY toolbox.
%
% fixedPos = [timeStart, traceStart], 1-based.
% Supported format codes: 1 IBM float, 2 int32, 3 int16, 5 IEEE float, 8 int8.

    info = getSEGYInfo_noToolbox(segyPath);
    dt_ms = info.dt_ms;

    assert(numel(fixedPos) == 2, 'fixedPos must be [timeStart, traceStart].');

    i0 = fixedPos(1);
    j0 = fixedPos(2);

    assert(i0 >= 1 && i0 <= info.ns - T + 1, ...
        'timeStart out of range: i0=%d, allowed 1..%d', i0, info.ns - T + 1);

    assert(j0 >= 1 && j0 <= info.ntr - W + 1, ...
        'traceStart out of range: j0=%d, allowed 1..%d', j0, info.ntr - W + 1);

    fid = fopen(segyPath, 'r', 'ieee-be');
    assert(fid > 0, 'Cannot open file: %s', segyPath);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

    patch = zeros(T, W, 'single');

    for jj = 1:W
        tr = j0 + (jj - 1);

        tr0   = 3600 + (tr - 1) * info.traceBytes;
        samp0 = tr0 + 240;

        fseek(fid, samp0 + (i0 - 1) * info.bytesPerSample, 'bof');

        switch info.formatCode
            case 5
                x = fread(fid, T, 'float32=>single');
            case 1
                raw = fread(fid, [4, T], 'uint8=>uint8');
                x = ibm2single_local(raw);
            case 2
                x = fread(fid, T, 'int32=>single');
            case 3
                x = fread(fid, T, 'int16=>single');
            case 8
                x = fread(fid, T, 'int8=>single');
        end

        if numel(x) ~= T
            error('Failed to read trace %d, read %d/%d samples.', tr, numel(x), T);
        end

        patch(:, jj) = x(:);
    end

    meta = info;
    meta.i0 = i0;
    meta.j0 = j0;
    meta.T = T;
    meta.W = W;
end

function x = ibm2single_local(raw4xN)
% Convert IBM 32-bit float bytes to IEEE single.

    b = raw4xN;
    signBit = bitshift(b(1,:), -7);
    signVal = 1 - 2 * single(signBit);

    exp16 = bitand(b(1,:), 127);
    E = single(exp16) - 64;

    frac = uint32(b(2,:));
    frac = bitor(bitshift(frac, 16), bitshift(uint32(b(3,:)), 8));
    frac = bitor(frac, uint32(b(4,:)));

    F = single(frac) / single(2^24);

    x = signVal .* F .* (16.^E);
    x = reshape(single(x), [], 1);
end
