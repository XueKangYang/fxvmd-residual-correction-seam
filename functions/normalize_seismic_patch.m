function out = normalize_seismic_patch(patch, scalePercentile)
%NORMALIZE_SEISMIC_PATCH Remove mean and scale a seismic patch robustly.

    if nargin < 2
        scalePercentile = 99;
    end

    out = double(patch);
    out = out - mean(out(:));
    s = prctile(abs(out(:)), scalePercentile);
    if s <= 0
        s = std(out(:)) + eps;
    end
    out = out ./ (s + eps);
end
