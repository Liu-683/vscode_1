clc;
clear;
%% 第八步

%
load("trainedUNet3D.mat");


%读取测试图像，并预处理
testImpath='C:\Users\chezhu\Desktop\vscode_unet3d_matlab\data\image\image2.tif';
testlapath='C:\Users\chezhu\Desktop\vscode_unet3d_matlab\data\label\label2.tif';
infoTest=imfinfo(testImpath);
H = infoTest(1).Height;
W = infoTest(1).Width;
numSlices=numel(infoTest);
testVol = zeros(H, W, numSlices, 'uint8');
for s = 1:numSlices
    testVol(:,:,s) = imread(testImpath, s);
end
testMaxVal = single(max(testVol(:)));
if testMaxVal > 0
    testVol = single(testVol) / testMaxVal;
else
    testVol = single(testVol);
end

testLbl = zeros(H, W, numSlices, 'uint8');
for s = 1:numSlices
    testLbl(:,:,s) = imread(testlapath, s);
end

fprintf('测试体积尺寸: [%d x %d x %d]\n', H, W, numSlices);

%% 切割分块预测
% 逐块预测并拼接结果
predVol = zeros(H, W, numSlices, 'uint8');

hStarts = computePatchStarts(H, patchSize(1));
wStarts = computePatchStarts(W, patchSize(2));
dStarts = computePatchStarts(numSlices, patchSize(3));

totalPatches = numel(hStarts) * numel(wStarts) * numel(dStarts);
pIdx = 0;

for hi = 1:numel(hStarts)
    for wi = 1:numel(wStarts)
        for di = 1:numel(dStarts)
            hs = hStarts(hi); he = hs + patchSize(1) - 1;
            ws = wStarts(wi); we = ws + patchSize(2) - 1;
            ds = dStarts(di); de = ds + patchSize(3) - 1;

            % 提取分块并添加通道维度
            patch = testVol(hs:he, ws:we, ds:de);
            patch = reshape(single(patch), [patchSize, 1]);

            % 使用 dlarray 进行预测
            dlPatch = dlarray(patch, 'SSSCB');
            dlPred  = predict(net, dlPatch);
            scores  = extractdata(dlPred);

            % 取 argmax 得到类别索引（沿通道维度）
            [~, classIdx] = max(scores, [], 4);
            classIdx = squeeze(classIdx);

            predVol(hs:he, ws:we, ds:de) = uint8(classIdx);

            pIdx = pIdx + 1;
            if mod(pIdx, 20) == 0
                fprintf('  推理进度: %d/%d 分块\n', pIdx, totalPatches);
            end
        end
    end
end
fprintf('推理完成。\n\n');

%% ==================== 第9步：计算评价指标 ====================
fprintf('=== 第9步：计算评价指标 ===\n');

% 转换为二值进行指标计算
predFG = (predVol == 2);     % 预测前景
gtFG   = (testLbl == 255);   % 真实前景

% Dice 系数
intersection = sum(predFG(:) & gtFG(:));
sumPred = sum(predFG(:));
sumGT   = sum(gtFG(:));
if (sumPred + sumGT) == 0
    diceScore = 1.0;  % 两者均为空，视为完美匹配
else
    diceScore = 2 * intersection / (sumPred + sumGT);
end

% 像素精度
predLblVis = zeros(size(predVol), 'uint8');
predLblVis(predVol == 2) = 255;
accuracy = sum(predLblVis(:) == testLbl(:)) / numel(testLbl);

% IoU（交并比）
unionCount = sum(predFG(:) | gtFG(:));
if unionCount == 0
    iouScore = 1.0;  % 两者均为空，视为完美匹配
else
    iouScore = intersection / unionCount;
end

fprintf('  Dice 系数:    %.4f\n', diceScore);
fprintf('  像素精度:     %.4f\n', accuracy);
fprintf('  前景 IoU:     %.4f\n\n', iouScore);

%多页tif
predTifPath = 'predicted_volume.tif';
if exist(predTifPath, 'file')
    delete(predTifPath);
end

for s = 1:numSlices
    sliceToWrite = predLblVis(:, :, s);
    if s == 1
        imwrite(sliceToWrite, predTifPath, 'Compression', 'none');
    else
        imwrite(sliceToWrite, predTifPath, 'WriteMode', 'append', 'Compression', 'none');
    end
end
%% ==================== 第10步：可视化分割结果 ====================
fprintf('=== 第10步：可视化分割结果 ===\n');

midSlice = round(numSlices / 2);

figure('Name', '二元语义分割结果', 'Position', [100 100 1500 400]);

subplot(1, 3, 1);
imagesc(testVol(:, :, midSlice));
colormap(gca, 'gray'); axis image off;
title('输入图像', 'FontSize', 14);

subplot(1, 3, 2);
imagesc(testLbl(:, :, midSlice));
colormap(gca, 'gray'); axis image off;
title('真实标签', 'FontSize', 14);

subplot(1, 3, 3);
imagesc(predLblVis(:, :, midSlice));
colormap(gca, 'gray'); axis image off;
title(sprintf('预测结果 (Dice=%.4f)', diceScore), 'FontSize', 14);

sgtitle('3D U-Net 二元语义分割结果', 'FontSize', 16);
saveas(gcf, 'segmentation_results.png');
fprintf('可视化结果已保存至 segmentation_results.png\n');

fprintf('\n=== 全部流程完成！ ===\n');