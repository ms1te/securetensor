% 读取 ECG 数据
data = readtable('ecg.csv');
ecg_data = table2array(data(:, 1:140)); % 提取前 140 列（ECG 数据点）

% 随机选取 1000 个患者
num_patients = 1000;
sample_indices = randperm(size(ecg_data, 1), num_patients);
selected_data = ecg_data(sample_indices, :);

% 初始化张量 (1000 × 110 × 140)
num_features = 110;
cluster_data = zeros(num_patients, num_features, 140);

% 针对每个时间点进行聚类
for i = 1:140
    % 聚类当前时间点的所有患者数据
    [cluster_indices, ~] = kmeans(selected_data(:, i), num_features);
    
    % 将聚类结果转换为 one-hot 编码
    % cluster_indices 是 (1000 × 1)，我们需要 (1000 × 110)
    one_hot = zeros(num_patients, num_features);
    for j = 1:num_patients
        one_hot(j, cluster_indices(j)) = 1;
    end
    
    % 将 one-hot 编码结果填入张量的第 i 个时间点
    cluster_data(:, :, i) = one_hot;
end

% 构造张量
ecg_tensor = cluster_data;

% 保存结果
save('ecg_tensor.mat', 'ecg_tensor');
disp('ECG tensor successfully created and saved.');


% 提示
% 执行完代码后，张量将以 .mat 文件保存（分别为 diabetes_tensor 和 ecg_tensor）。
% 你可以用 `load('diabetes_tensor.mat')` 或 `load('ecg_tensor.mat')` 加载结果。


% 清空工作空间
clear; clc;

% 加载数据文件
% 确保文件位于工作目录下，或指定完整路径
data = readtable('diabetes_binary_health_indicators_BRFSS2015.csv');

% Step 1: 随机选择 1000 个样本
num_samples = 1000; % 样本数量
sample_indices = randperm(height(data), num_samples); % 随机选取样本索引
selected_data = data(sample_indices, :);

% Step 2: 随机选择 20 个生理指标和 24 个习惯特征
num_physiological = 20; % 生理指标数量
num_habitual = 24; % 习惯特征数量

% 获取所有特征的列名（去掉目标变量列）
feature_columns = data.Properties.VariableNames(2:end);

% 获取总特征数量
total_features = length(feature_columns); % 21 个特征

% 确保选择的特征数量不超过总特征数量
num_physiological = min(20, total_features); % 生理指标最多 20 个
num_habitual = min(24, total_features);     % 习惯特征最多 24 个

% 随机选取特征索引
physiological_indices = randperm(total_features, num_physiological);
habitual_indices = randperm(total_features, num_habitual);

% 提取对应的列
physiological_features = selected_data(:, physiological_indices);
habitual_features = selected_data(:, habitual_indices);

disp('Selected physiological and habitual features successfully.');


% Step 3: 构建张量
% 初始化张量
diabetes_tensor = zeros(num_samples, num_physiological, num_habitual);

% 填充张量
for i = 1:num_samples
    % 提取第 i 个样本的特征值
    physiological_values = table2array(physiological_features(i, :));
    habitual_values = table2array(habitual_features(i, :));
    
    % 将值填入张量
    diabetes_tensor(i, :, :) = physiological_values' * habitual_values;
end

% Step 4: 保存张量到文件
save('diabetes_tensor.mat', 'diabetes_tensor');

% 输出完成消息
disp('Diabetes tensor successfully created and saved as diabetes_tensor.mat');


