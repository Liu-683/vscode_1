function starts = computePatchStarts(dimSize, patchDim)
%COMPUTEPATCHSTARTS 计算分块的起始位置
%   对于给定的维度大小和分块大小，计算各分块的起始索引。
%   如果最后一个分块超出边界，则调整其起始位置使其贴紧末端。
%   注意：边界处的最后一个分块可能与前一个分块有部分重叠，
%   这是为了确保所有体素都被完整大小的分块覆盖。
    if dimSize < patchDim
        error('维度大小 (%d) 小于分块大小 (%d)，无法提取分块。', dimSize, patchDim);
    end
    starts = 1:patchDim:dimSize;
    if starts(end) + patchDim - 1 > dimSize
        starts(end) = dimSize - patchDim + 1;
    end
    starts = unique(starts);
end