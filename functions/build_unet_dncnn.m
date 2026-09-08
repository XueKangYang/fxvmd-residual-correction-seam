function lgraph = build_unet_dncnn(depth, baseNumFilters, inputSize)
%BUILD_UNET_DNCNN Build a two-channel U-Net x DnCNN style residual network.
%
% Input:
%   depth          - encoder/decoder depth
%   baseNumFilters - number of filters in the first encoder stage
%   inputSize      - [time, trace, channel], here channel = 2
%
% Output:
%   lgraph         - MATLAB layer graph for regression training

    inputLayer = imageInputLayer(inputSize, 'Name','input','Normalization','none');
    lgraph = layerGraph(inputLayer);
    encoderNames = cell(depth,1);

    %% -------- Encoder --------
    for d = 1:depth
        nf = baseNumFilters * 2^(d-1);

        blk = [
            convolution2dLayer(3, nf, 'Padding','same', 'Name', sprintf('enc%d_conv1',d))
            batchNormalizationLayer('Name', sprintf('enc%d_bn1',d))
            reluLayer('Name', sprintf('enc%d_relu1',d))

            convolution2dLayer(3, nf, 'Padding','same', 'Name', sprintf('enc%d_conv2',d))
            batchNormalizationLayer('Name', sprintf('enc%d_bn2',d))
            reluLayer('Name', sprintf('enc%d_relu2',d))

            convolution2dLayer(3, nf, 'Padding','same', 'Name', sprintf('enc%d_conv3',d))
            batchNormalizationLayer('Name', sprintf('enc%d_bn3',d))
            reluLayer('Name', sprintf('enc%d_relu3',d))
        ];

        pool = maxPooling2dLayer([2 1], 'Stride',[2 1], 'Name', sprintf('pool%d', d));

        lgraph = addLayers(lgraph, blk);
        lgraph = addLayers(lgraph, pool);

        if d == 1
            lgraph = connectLayers(lgraph, 'input', sprintf('enc%d_conv1',d));
        else
            lgraph = connectLayers(lgraph, sprintf('pool%d', d-1), sprintf('enc%d_conv1',d));
        end

        lgraph = connectLayers(lgraph, sprintf('enc%d_relu3', d), sprintf('pool%d', d));
        encoderNames{d} = sprintf('enc%d_relu3', d);
    end

    %% -------- Bottleneck --------
    bnf = baseNumFilters * 2^depth;

    b_blk = [
        convolution2dLayer(3, bnf, 'Padding','same','Name','b_conv1')
        batchNormalizationLayer('Name','b_bn1')
        reluLayer('Name','b_relu1')

        convolution2dLayer(3, bnf, 'Padding','same','Name','b_conv2')
        batchNormalizationLayer('Name','b_bn2')
        reluLayer('Name','b_relu2')

        convolution2dLayer(3, bnf, 'Padding','same','Name','b_conv3')
        batchNormalizationLayer('Name','b_bn3')
        reluLayer('Name','b_relu3')
    ];

    lgraph = addLayers(lgraph, b_blk);
    lgraph = connectLayers(lgraph, sprintf('pool%d', depth), 'b_conv1');

    %% -------- Decoder --------
    for d = depth:-1:1
        nf = baseNumFilters * 2^(d-1);
        upName = sprintf('up%d', d);

        up = [
            transposedConv2dLayer([2 1], nf, 'Stride',[2 1], ...
                'Cropping','same', 'Name', sprintf('%s_trans', upName))
            batchNormalizationLayer('Name', sprintf('%s_bn', upName))
            reluLayer('Name', sprintf('%s_relu', upName))
        ];

        concat = concatenationLayer(3,2,'Name', sprintf('%s_concat', upName));

        decBlk = [
            convolution2dLayer(3, nf, 'Padding','same', 'Name', sprintf('dec%d_conv1', d))
            batchNormalizationLayer('Name', sprintf('dec%d_bn1', d))
            reluLayer('Name', sprintf('dec%d_relu1', d))

            convolution2dLayer(3, nf, 'Padding','same', 'Name', sprintf('dec%d_conv2', d))
            batchNormalizationLayer('Name', sprintf('dec%d_bn2', d))
            reluLayer('Name', sprintf('dec%d_relu2', d))

            convolution2dLayer(3, nf, 'Padding','same', 'Name', sprintf('dec%d_conv3', d))
            batchNormalizationLayer('Name', sprintf('dec%d_bn3', d))
            reluLayer('Name', sprintf('dec%d_relu3', d))
        ];

        lgraph = addLayers(lgraph, up);
        lgraph = addLayers(lgraph, concat);
        lgraph = addLayers(lgraph, decBlk);

        if d == depth
            lgraph = connectLayers(lgraph, 'b_relu3', sprintf('%s_trans', upName));
        else
            lgraph = connectLayers(lgraph, sprintf('dec%d_relu3', d+1), sprintf('%s_trans', upName));
        end

        lgraph = connectLayers(lgraph, sprintf('%s_relu', upName), sprintf('%s_concat/in1', upName));
        lgraph = connectLayers(lgraph, encoderNames{d}, sprintf('%s_concat/in2', upName));
        lgraph = connectLayers(lgraph, sprintf('%s_concat', upName), sprintf('dec%d_conv1', d));
    end

    %% -------- Output head --------
    finalConv = convolution2dLayer(1, 1, 'Padding','same','Name','final_conv');
    regLayer  = regressionLayer('Name','regression');

    lgraph = addLayers(lgraph, finalConv);
    lgraph = addLayers(lgraph, regLayer);

    lgraph = connectLayers(lgraph, 'dec1_relu3', 'final_conv');
    lgraph = connectLayers(lgraph, 'final_conv', 'regression');
end
