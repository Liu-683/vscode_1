%% main.m - 基于3D U-Net的二元语义分割
%
% 功能说明：
%   读取 data 目录中的3D多页TIF图像和标签数据，
%   使用3D U-Net模型进行二元语义分割训练和预测。
%   训练时将数据分块(patch)处理并保存到磁盘，以防止内存不足。
%
% 运行要求：
%   - MATLAB R2024a 或更高版本
%   - Deep Learning Toolbox
%   - Computer Vision Toolbox
%   - Image Processing Toolbox
%
% 数据结构：
%   data/
%     image/  -> image1.tif, image2.tif, image3.tif  (uint8, 多页3D体积)
%     label/  -> label1.tif, label2.tif, label3.tif  (uint8, 0和255二值标签)

clear; clc; close all;

%% ==================== 参数配置 ====================
% 分块参数
patchSize     = [64 64 64];       % 3D分块大小 [H W D]

% 网络参数
numClasses    = 2;                 % 类别数：背景 + 前景
numChannels   = 1;                 % 输入通道数（灰度图像）
encoderDepth  = 3;                 % U-Net编码器深度

% 训练参数
initialLR     = 1e-3;             % 初始学习率
maxEpochs     = 10;               % 最大训练轮数
miniBatchSize = 2;                 % 每批次样本数（3D体积占用大量显存，建议根据GPU内存调整）
valRatio      = 0.2;              % 验证集比例

% 类别信息
classNames  = ["background", "foreground"];
labelValues = [0, 255];            % 标签TIF文件中的像素值

% 路径设置
dataDir  = fullfile(pwd, 'data');
imageDir = fullfile(dataDir, 'image');
labelDir = fullfile(dataDir, 'label');
patchDir = fullfile(tempdir, 'unet3d_patches');

% 文件列表（图像和标签一一对应）
imageFiles = {'image1.tif', 'image2.tif', 'image3.tif'};
labelFiles = {'label1.tif', 'label2.tif', 'label3.tif'};
numVolumes = numel(imageFiles);

%% ==================== 第1步：读取数据并提取分块 ====================
fprintf('=== 第1步：读取数据并提取分块 ===\n');

% 创建分块保存目录
imgPatchDir = fullfile(patchDir, 'images');
lblPatchDir = fullfile(patchDir, 'labels');
if ~exist(imgPatchDir, 'dir'), mkdir(imgPatchDir); end
if ~exist(lblPatchDir, 'dir'), mkdir(lblPatchDir); end

patchCount = 0;

for v = 1:numVolumes
    fprintf('正在处理第 %d/%d 个体积数据...\n', v, numVolumes);

    % --- 读取多页TIF图像 ---
    imgPath = fullfile(imageDir, imageFiles{v});
    infoImg = imfinfo(imgPath);
    numSlices = numel(infoImg);
    H = infoImg(1).Height;
    W = infoImg(1).Width;

    vol = zeros(H, W, numSlices, 'uint8');
    for s = 1:numSlices
        vol(:,:,s) = imread(imgPath, s);
    end

    % --- 读取多页TIF标签 ---
    lblPath = fullfile(labelDir, labelFiles{v});
    lbl = zeros(H, W, numSlices, 'uint8');
    for s = 1:numSlices
        lbl(:,:,s) = imread(lblPath, s);
    end

    fprintf('  图像尺寸: [%d x %d x %d], 标签尺寸: [%d x %d x %d]\n', ...
        size(vol, 1), size(vol, 2), size(vol, 3), ...
        size(lbl, 1), size(lbl, 2), size(lbl, 3));

    % --- 将图像归一化到 [0, 1] ---
    maxVal = single(max(vol(:)));
    if maxVal > 0
        vol = single(vol) / maxVal;
    else
        vol = single(vol);
    end

    % --- 将标签转换为类别索引：0->1(背景), 255->2(前景) ---
    lblIdx = zeros(size(lbl), 'uint8');
    lblIdx(lbl == 0)   = 1;  % 背景
    lblIdx(lbl == 255) = 2;  % 前景

    % --- 提取非重叠分块并保存到磁盘 ---
    [vH, vW, vD] = size(vol);

    hStarts = computePatchStarts(vH, patchSize(1));
    wStarts = computePatchStarts(vW, patchSize(2));
    dStarts = computePatchStarts(vD, patchSize(3));

    for hi = 1:numel(hStarts)
        for wi = 1:numel(wStarts)
            for di = 1:numel(dStarts)
                hs = hStarts(hi); he = hs + patchSize(1) - 1;
                ws = wStarts(wi); we = ws + patchSize(2) - 1;
                ds = dStarts(di); de = ds + patchSize(3) - 1;

                patchCount = patchCount + 1;

                % 提取分块，图像添加通道维度: [H W D] -> [H W D 1]
                imgPatch = reshape(vol(hs:he, ws:we, ds:de), [patchSize, 1]); %#ok<NASGU>
                lblPatch = lblIdx(hs:he, ws:we, ds:de);                       %#ok<NASGU>

                % 保存到磁盘（节省内存）
                save(fullfile(imgPatchDir, sprintf('img_%05d.mat', patchCount)), 'imgPatch');
                save(fullfile(lblPatchDir, sprintf('lbl_%05d.mat', patchCount)), 'lblPatch');
            end
        end
    end

    % 释放当前体积数据以节省内存
    clear vol lbl lblIdx;
    fprintf('  已从该体积提取分块。\n');
end

fprintf('分块提取完成，共 %d 个分块。\n\n', patchCount);

%% ==================== 第2步：划分训练集和验证集 ====================
fprintf('=== 第2步：划分训练集和验证集 ===\n');

rng(42);  % 设置随机种子以确保可重复性
idx = randperm(patchCount);
numVal   = round(valRatio * patchCount);
numTrain = patchCount - numVal;

trainIdx = idx(1:numTrain);
valIdx   = idx(numTrain+1:end);

fprintf('训练分块数: %d, 验证分块数: %d\n\n', numTrain, numVal);

% 构建训练集和验证集的文件路径列表
trainImgFiles = arrayfun(@(k) fullfile(imgPatchDir, sprintf('img_%05d.mat', k)), ...
    trainIdx, 'UniformOutput', false);
trainLblFiles = arrayfun(@(k) fullfile(lblPatchDir, sprintf('lbl_%05d.mat', k)), ...
    trainIdx, 'UniformOutput', false);
valImgFiles = arrayfun(@(k) fullfile(imgPatchDir, sprintf('img_%05d.mat', k)), ...
    valIdx, 'UniformOutput', false);
valLblFiles = arrayfun(@(k) fullfile(lblPatchDir, sprintf('lbl_%05d.mat', k)), ...
    valIdx, 'UniformOutput', false);

%% ==================== 第3步：创建数据存储 ====================
fprintf('=== 第3步：创建数据存储 ===\n');

% 定义读取函数
readImgFcn = @(f) loadMatField(f, 'imgPatch');
readLblFcn = @(f) categorical(loadMatField(f, 'lblPatch'), [1 2], classNames);

% 创建训练数据存储
dsTrainImg = fileDatastore(trainImgFiles, 'ReadFcn', readImgFcn);
dsTrainLbl = fileDatastore(trainLblFiles, 'ReadFcn', readLblFcn);
dsTrain    = combine(dsTrainImg, dsTrainLbl);

% 创建验证数据存储
dsValImg = fileDatastore(valImgFiles, 'ReadFcn', readImgFcn);
dsValLbl = fileDatastore(valLblFiles, 'ReadFcn', readLblFcn);
dsVal    = combine(dsValImg, dsValLbl);

fprintf('数据存储创建完成。\n\n');

%% ==================== 第4步：构建3D U-Net网络 ====================
fprintf('=== 第4步：构建3D U-Net网络 ===\n');

inputSize = [patchSize, numChannels]; % [64 64 64 1]

% 使用 unet3d 函数直接创建 3D U-Net dlnetwork（需要 Computer Vision Toolbox）
net = unet3d(inputSize, numClasses, 'EncoderDepth', encoderDepth);

fprintf('3D U-Net 网络构建完成，共 %d 层。\n\n', numel(net.Layers));

%% ==================== 第5步：配置训练选项 ====================
fprintf('=== 第5步：配置训练选项 ===\n');

% 每个epoch验证一次（可根据需要调整 valFreq 的值）
valFreq = max(1, floor(numTrain / miniBatchSize));

options = trainingOptions('adam', ...
    'InitialLearnRate', initialLR, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', dsVal, ...
    'ValidationFrequency', valFreq, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 15, ...
    'L2Regularization', 1e-4, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 5, ...
    'ExecutionEnvironment', 'auto');

fprintf('训练选项配置完成。\n\n');

%% ==================== 第6步：训练网络 ====================
fprintf('=== 第6步：开始训练 ===\n');
fprintf('正在使用 trainnet 进行训练，请等待...\n');

[net, info] = trainnet(dsTrain, net, "crossentropy", options);

fprintf('训练完成！\n\n');

%% ==================== 第7步：保存模型 ====================
fprintf('=== 第7步：保存训练好的模型 ===\n');

%save('trainedUNet3D.mat', 'net', 'info', 'classNames', 'patchSize');
fprintf('模型已保存至 trainedUNet3D.mat\n\n');


%% ==================== 辅助函数 ====================

function data = loadMatField(filename, fieldName)
%LOADMATFIELD 从 .mat 文件中加载指定字段
    s = load(filename, fieldName);
    data = s.(fieldName);
end
