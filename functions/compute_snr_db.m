function snrVal = compute_snr_db(clean, estimate)
%COMPUTE_SNR_DB Compute SNR between a clean reference and an estimate.

    clean = double(clean);
    estimate = double(estimate);
    sigp = mean(clean(:).^2);
    errp = mean((estimate(:) - clean(:)).^2);
    snrVal = 10 * log10(sigp / max(errp, eps));
end
