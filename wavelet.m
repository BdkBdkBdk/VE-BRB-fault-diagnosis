function [TrainData, medianStats] = wavelet(filename, waveletType)
    % wavelet.m
    % 功能: 读取指定数据文件，进行小波变换并提取能量特征及中位数统计值
    %
    % 输入参数:
    %   - filename: 数据文件名称 (如 'test_261.txt')
    %   - waveletType: 小波基函数 (默认 'db4')
    %
    % 输出参数:
    %   - TrainData: 包含 [DE能量, FE能量, 标签] 的矩阵
    %   - medianStats: 结构体，含四组能量的中位数（cA_DE、cD_DE、cA_FE、cD_FE）

    % 参数检查
    if nargin < 1
        error('请提供数据文件名');
    end
    if nargin < 2
        waveletType = 'db4';
    end

    % 加载数据
    try
        TestData = load(filename);
    catch
        error('无法加载文件: %s', filename);
    end

    if size(TestData, 2) < 3
        error('数据文件应至少包含三列 (DE, FE, BS)');
    end

    % 提取 DE、FE、BS 信号
    DE = TestData(:, 1);
    FE = TestData(:, 2);
    BS = TestData(:, 3);

    % 小波变换
    [cA_DE, cD_DE] = dwt(DE, waveletType);
    [cA_FE, cD_FE] = dwt(FE, waveletType);

    % 计算各分量能量
    cA_DE_energy = abs(cA_DE).^2;
    cD_DE_energy = abs(cD_DE).^2;
    cA_FE_energy = abs(cA_FE).^2;
    cD_FE_energy = abs(cD_FE).^2;

    % 合并能量
%     wavelet_energy_DE = cA_DE_energy + cD_DE_energy;
%     wavelet_energy_FE = cA_FE_energy + cD_FE_energy;

    % 截取最短长度
    num_rows = min([length(cA_DE_energy), length(cD_DE_energy), length(BS)]);
    TrainData = [cA_DE_energy(1:num_rows), cD_DE_energy(1:num_rows), BS(1:num_rows)];

    % 统计能量中位数
    medianStats.cA_DE_median = median(cA_DE_energy);
    medianStats.cD_DE_median = median(cD_DE_energy);
    medianStats.cA_FE_median = median(cA_FE_energy);
    medianStats.cD_FE_median = median(cD_FE_energy);
end
