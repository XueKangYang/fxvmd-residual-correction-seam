function info = getSEGYInfo_noToolbox(segyPath)
%GETSEGYINFO_NOTOOLBOX Read basic SEGY metadata without the MATLAB SEGY toolbox.

    fid = fopen(segyPath, 'r', 'ieee-be');
    assert(fid > 0, 'Cannot open file: %s', segyPath);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

    d = dir(segyPath);
    fileBytes = d.bytes;
    assert(fileBytes > 3600, 'File too small to be SEGY. bytes=%d', fileBytes);

    fseek(fid, 3200, 'bof');
    bin = fread(fid, 400, 'uint8=>uint8');
    assert(numel(bin) == 400, 'Failed to read 400-byte SEGY binary header.');

    be_u16 = @(b1,b2) double(uint16(b1) * 256 + uint16(b2));

    dt_us = be_u16(bin(17), bin(18));
    ns    = be_u16(bin(21), bin(22));
    fmt   = be_u16(bin(25), bin(26));

    if dt_us <= 0 || dt_us > 1e6
        warning('Invalid dt in SEGY header; using 1000 us.');
        dt_us = 1000;
    end

    if ns <= 0 || ns > 5e6
        error('Invalid sample number read from header: %g.', ns);
    end

    switch fmt
        case 1
            bytesPerSample = 4;  % IBM float
        case 2
            bytesPerSample = 4;  % int32
        case 3
            bytesPerSample = 2;  % int16
        case 5
            bytesPerSample = 4;  % IEEE float
        case 8
            bytesPerSample = 1;  % int8
        otherwise
            error('Unsupported SEGY format code = %d.', fmt);
    end

    traceBytes = 240 + ns * bytesPerSample;
    dataBytes  = fileBytes - 3600;
    ntr        = floor(dataBytes / traceBytes);

    info = struct();
    info.ns = ns;
    info.ntr = ntr;
    info.dt_us = dt_us;
    info.dt_ms = dt_us / 1000;
    info.dt_s = dt_us / 1e6;
    info.formatCode = fmt;
    info.bytesPerSample = bytesPerSample;
    info.traceBytes = traceBytes;
    info.fileBytes = fileBytes;
end
