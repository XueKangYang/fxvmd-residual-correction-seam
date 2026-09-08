function noisy = add_awgn_measured(clean, snr_db)
%ADD_AWGN_MEASURED Add white Gaussian noise according to measured signal power.
% This avoids dependency on the Communications Toolbox awgn function.

    clean = double(clean);
    signalPower = mean(clean(:).^2);
    noisePower = signalPower / (10^(snr_db/10));
    noise = sqrt(noisePower) * randn(size(clean));
    noisy = clean + noise;
end
