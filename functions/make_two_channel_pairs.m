function [X2ch, Y, normInfo] = make_two_channel_pairs(noisy3D, clean3D, fxvmd3D)
%MAKE_TWO_CHANNEL_PAIRS Construct [R_fx, F] input and true residual target.
%
% Inputs are [T, W, N]. Output X2ch is [T, W, 2, N], Y is [T, W, 1, N].

    [T, W, N] = size(noisy3D);

    RfxNorm = zeros(T, W, N);
    FNorm   = zeros(T, W, N);
    RtrueNorm = zeros(T, W, N);

    mu = zeros(1, N);
    sigma = zeros(1, N);

    for i = 1:N
        D = noisy3D(:,:,i);
        C = clean3D(:,:,i);
        F = fxvmd3D(:,:,i);

        Rfx = D - F;
        Rtrue = D - C;

        mu(i) = mean(Rfx(:));
        sigma(i) = std(Rfx(:)) + eps;

        RfxNorm(:,:,i) = (Rfx - mu(i)) ./ sigma(i);
        FNorm(:,:,i) = (F - mu(i)) ./ sigma(i);
        RtrueNorm(:,:,i) = (Rtrue - mu(i)) ./ sigma(i);
    end

    X2ch = cat(3, reshape(RfxNorm, T, W, 1, N), reshape(FNorm, T, W, 1, N));
    Y = reshape(RtrueNorm, T, W, 1, N);

    normInfo = struct('mu', mu, 'sigma', sigma);
end
