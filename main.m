%% Third Blood for IEEE Access
clc
clear all
global L
global M
global KK
%global KKK


global AllData
global TrainData
L = 25;
M = 2;
N = 3;
%KKK=zeros(503,3);
%% ==========the training data and testing data input =====
[BRB_I ] = generator (1);

Train=load('train.txt');  
% 调用 wavelet 函数处理五个文件
TrainData_100 = wavelet('train_100_less.txt', 'db4');
TrainData_212 = wavelet('train_212_less.txt', 'db4');
TrainData_261 = wavelet('train_261_less.txt', 'db4');
% 假设这些变量都存在且是数值矩阵
% 合并所有数据
TrainData = [TrainData_100; TrainData_212; TrainData_261;];
dlmwrite('Trainparse.txt', TrainData, 'delimiter', '\t', 'precision', 6);
% 调用 wavelet 函数处理三个数据文件
[FullData_100, stats_100] = wavelet('FullData_100.txt', 'db4');
[FullData_212, stats_212] = wavelet('FullData_212.txt', 'db4');
[FullData_261, stats_261] = wavelet('FullData_261.txt', 'db4');

% 显示中位数统计信息
disp('--- Stats for FullData_100 ---');
disp(stats_100);

disp('--- Stats for FullData_212 ---');
disp(stats_212);

disp('--- Stats for FullData_261 ---');
disp(stats_261);

FullData = wavelet('FullData_212.txt', 'db4');


RuleW_I = 1 * ones(L);                  % initial value for rule weight
AttributeW_I = 1 * ones(M);             % initial value for attribute weight
%% =======the optimization model==========
for k=1:L            %L条规则
    for n=1:N        %%%%N条属性
        x0((k-1)*N+n)=BRB_I(k,n);   % belief degree
    end
end
for k=1:L
    x0(L*N + k) = RuleW_I (k);      % rule weight
end
for k=1:M
    x0(L*N + L + k) = AttributeW_I (k);   %% attribute weight
end

x0 = x0';
lb=zeros(L*N+L+M ,1);
ub=ones(L*N+L+M ,1);
Aeq=zeros(L+L+M, L*N+L+M);

for k=1:L
    for n=1:N
        Aeq(k,(k-1)*N+n)=1;
    end
end

beq=ones(L+L+M,1); k
for i =1:L+M
    beq(L+i) = 0;
end


G =200;
A = [];b = [];
% 初始化准确度数组
predict_status = [0,0,0];
numCases = N;           %定义分类数量
numTests = 5000;
accuracyArray = zeros(numCases, numTests);
Xbest = cmaes(x0,G,Aeq,beq,ub,lb);

    


% 创建一个1x15000的矩阵 predicted_labels，用于存储预测状态
predicted_labels = zeros(15000, 1);
Num = 0;
successful_status = 0;
for t = 1:3
    AllData = zeros(503, 3);
    if t == 1     
        FullData = FullData_100;
    elseif t == 2
        FullData = FullData_212;
    elseif t == 3
        FullData = FullData_261;
    end
    AllData(1:503, 1) = FullData(1:503, 1); % 写入第一列
    AllData(1:503, 2) = FullData(1:503, 2); % 写入第二列
    AllData(1:503, 3) = FullData(1:503, 3); % 写入第三列
    for m=1:numTests
        AllData = zeros(503, 3);
        AllData(1:503, 3) = FullData(1:503, 3); % 写入第三列
        AllData(1:503, 1) = FullData(m * 6+504:m * 6+1006, 1); % 写入第一列
        AllData(1:503, 2) = FullData(m * 6+504:m * 6+1006, 2); % 写入第二列
        MSE(m)=fun_test(Xbest);
        Z(:,m)=KK';
    %     Z1(:,m)=KK';
    
          %四舍五入 
        predictions = Z(:,m);
        predictions(predictions <= 1.9) = 1;
        predictions(predictions > 1.9 & predictions <= 2.4) = 2;
        predictions(predictions > 2.4 & predictions <= 3.5) = 3;
        predictions(predictions > 3.5 & predictions <= 4.5) = 4;
        predictions(predictions > 4.5) = 5;
    %     edges = [1.9, 2.4, 3.5, 4.5, Inf]; % 定义区间边界
    %     predictions = discretize(Z(:,m), edges); % 根据区间边界将数据分类
        Z(:,m) = predictions;
        % 创建一个新的图形窗口
    %     figure;
    %     k = 1:503;
    %     plot(k, AllData(k, 3), k, Z(k),'.');
    % 计算各个状态的分布
        status_1_count = sum(Z(:, m) == 1);
        status_2_count = sum(Z(:, m) == 2);
        status_3_count = sum(Z(:, m) == 3);
        
        % 记录分布情况
        status_distribution(m + (t-1) * 5000, :) = [status_1_count, status_2_count, status_3_count];
    
    % 计算准确率
        total_samples = size(AllData, 1);
        for i = 1:numCases
            % 计算每种情况的分布
            correct_predictions = sum(Z(:,m) == i);
            accuracy = correct_predictions / total_samples * 100;
            % 将准确度存入数组
            accuracyArray(i, m) = accuracy;
        end
        % 最后判断

        if status_1_count / total_samples * 100 > 59
            fprintf('测试 %d: 状态1的比例超过59%%，预测为状态1。', m);
            predict_status(1) = predict_status(1) + 1;
            if AllData(1,3) == 1
                fprintf('预测成功√√√\n');
                successful_status = successful_status + 1;
                
            else
                fprintf('预测失败×××\n');
            end
            Num = Num+1;
            predicted_labels(Num) = 1;
        elseif status_2_count / total_samples * 100 > 50
            fprintf('测试 %d: 状态2的比例超过50%%，预测为状态2。', m);
            predict_status(2) = predict_status(2) + 1;
            if AllData(1,3) == 2
                fprintf('预测成功√√√\n');
                successful_status = successful_status + 1;
            else
                fprintf('预测失败×××\n');
            end
            Num = Num+1;
            predicted_labels(Num) = 2;
        elseif status_3_count / total_samples * 100 > 12
            fprintf('测试 %d: 状态3的比例超过12%%，预测为状态3。', m);
            predict_status(3) = predict_status(3) + 1;
            if AllData(1,3) == 3
                fprintf('预测成功√√√\n');
                successful_status = successful_status + 1;
            else
                fprintf('预测失败×××\n');
            end
            Num = Num+1;
            predicted_labels(Num) = 3;
        else
            fprintf('测试 %d: 自动归类。\n', m);
            predict_status(3) = predict_status(3) + 1;
            if AllData(1,3) == 3
                fprintf('预测成功√√√\n');
                successful_status = successful_status + 1;
            else
                fprintf('预测失败×××\n');
            end
            Num = Num+1;
            predicted_labels(Num) = 3;
        end

    end


% ————————————————————————————————柱状图———————————————————————————————— %
% 绘制准确率对比柱状图
figure;

% 绘制柱状图，展示状态1、状态2、状态3的准确率
bar(accuracyArray', 'grouped');  % 'grouped' 使不同状态并列显示

% 添加图表标签
xlabel('Test Run');
ylabel('Accuracy (%)');
title('Accuracy Comparison Across Different States');

% 添加图例
legend('State 1', 'State 2', 'State 3');

% 添加网格线
grid on;
% ————————————————————————————————柱状图———————————————————————————————— %



end
total_tests = numTests * 3;  % 总的测试次数
success_rate = (successful_status / total_tests) * 100;  % 计算成功率

% 输出总测试次数、成功预测次数以及成功率
fprintf('在 %d 次测试中，共成功预测 %d 次。\n', total_tests, successful_status);
fprintf('命中成功率为 %.2f%%。\n', success_rate);  % 输出成功率
