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
% ���� wavelet ������������ļ�
TrainData_100 = wavelet('train_100_less.txt', 'db4');
TrainData_212 = wavelet('train_212_less.txt', 'db4');
TrainData_261 = wavelet('train_261_less.txt', 'db4');
% ������Щ����������������ֵ����
% �ϲ���������
TrainData = [TrainData_100; TrainData_212; TrainData_261;];
dlmwrite('Trainparse.txt', TrainData, 'delimiter', '\t', 'precision', 6);
% ���� wavelet �����������������ļ�
[FullData_100, stats_100] = wavelet('FullData_100.txt', 'db4');
[FullData_212, stats_212] = wavelet('FullData_212.txt', 'db4');
[FullData_261, stats_261] = wavelet('FullData_261.txt', 'db4');

% ��ʾ��λ��ͳ����Ϣ
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
for k=1:L            %L������
    for n=1:N        %%%%N������
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
% ��ʼ��׼ȷ������
predict_status = [0,0,0];
numCases = N;           %�����������
numTests = 5000;
accuracyArray = zeros(numCases, numTests);
Xbest = cmaes(x0,G,Aeq,beq,ub,lb);

    


% ����һ��1x15000�ľ��� predicted_labels�����ڴ洢Ԥ��״̬
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
    AllData(1:503, 1) = FullData(1:503, 1); % д���һ��
    AllData(1:503, 2) = FullData(1:503, 2); % д��ڶ���
    AllData(1:503, 3) = FullData(1:503, 3); % д�������
    for m=1:numTests
        AllData = zeros(503, 3);
        AllData(1:503, 3) = FullData(1:503, 3); % д�������
        AllData(1:503, 1) = FullData(m * 6+504:m * 6+1006, 1); % д���һ��
        AllData(1:503, 2) = FullData(m * 6+504:m * 6+1006, 2); % д��ڶ���
        MSE(m)=fun_test(Xbest);
        Z(:,m)=KK';
    %     Z1(:,m)=KK';
    
          %�������� 
        predictions = Z(:,m);
        predictions(predictions <= 1.9) = 1;
        predictions(predictions > 1.9 & predictions <= 2.4) = 2;
        predictions(predictions > 2.4 & predictions <= 3.5) = 3;
        predictions(predictions > 3.5 & predictions <= 4.5) = 4;
        predictions(predictions > 4.5) = 5;
    %     edges = [1.9, 2.4, 3.5, 4.5, Inf]; % ��������߽�
    %     predictions = discretize(Z(:,m), edges); % ��������߽罫���ݷ���
        Z(:,m) = predictions;
        % ����һ���µ�ͼ�δ���
    %     figure;
    %     k = 1:503;
    %     plot(k, AllData(k, 3), k, Z(k),'.');
    % �������״̬�ķֲ�
        status_1_count = sum(Z(:, m) == 1);
        status_2_count = sum(Z(:, m) == 2);
        status_3_count = sum(Z(:, m) == 3);
        
        % ��¼�ֲ����
        status_distribution(m + (t-1) * 5000, :) = [status_1_count, status_2_count, status_3_count];
    
    % ����׼ȷ��
        total_samples = size(AllData, 1);
        for i = 1:numCases
            % ����ÿ������ķֲ�
            correct_predictions = sum(Z(:,m) == i);
            accuracy = correct_predictions / total_samples * 100;
            % ��׼ȷ�ȴ�������
            accuracyArray(i, m) = accuracy;
        end
        % ����ж�

        if status_1_count / total_samples * 100 > 59
            fprintf('���� %d: ״̬1�ı�������59%%��Ԥ��Ϊ״̬1��', m);
            predict_status(1) = predict_status(1) + 1;
            if AllData(1,3) == 1
                fprintf('Ԥ��ɹ��̡̡�\n');
                successful_status = successful_status + 1;
                
            else
                fprintf('Ԥ��ʧ�ܡ�����\n');
            end
            Num = Num+1;
            predicted_labels(Num) = 1;
        elseif status_2_count / total_samples * 100 > 50
            fprintf('���� %d: ״̬2�ı�������50%%��Ԥ��Ϊ״̬2��', m);
            predict_status(2) = predict_status(2) + 1;
            if AllData(1,3) == 2
                fprintf('Ԥ��ɹ��̡̡�\n');
                successful_status = successful_status + 1;
            else
                fprintf('Ԥ��ʧ�ܡ�����\n');
            end
            Num = Num+1;
            predicted_labels(Num) = 2;
        elseif status_3_count / total_samples * 100 > 12
            fprintf('���� %d: ״̬3�ı�������12%%��Ԥ��Ϊ״̬3��', m);
            predict_status(3) = predict_status(3) + 1;
            if AllData(1,3) == 3
                fprintf('Ԥ��ɹ��̡̡�\n');
                successful_status = successful_status + 1;
            else
                fprintf('Ԥ��ʧ�ܡ�����\n');
            end
            Num = Num+1;
            predicted_labels(Num) = 3;
        else
            fprintf('���� %d: �Զ����ࡣ\n', m);
            predict_status(3) = predict_status(3) + 1;
            if AllData(1,3) == 3
                fprintf('Ԥ��ɹ��̡̡�\n');
                successful_status = successful_status + 1;
            else
                fprintf('Ԥ��ʧ�ܡ�����\n');
            end
            Num = Num+1;
            predicted_labels(Num) = 3;
        end

    end


% ������������������������������������������������������������������״ͼ���������������������������������������������������������������� %
% ����׼ȷ�ʶԱ���״ͼ
figure;

% ������״ͼ��չʾ״̬1��״̬2��״̬3��׼ȷ��
bar(accuracyArray', 'grouped');  % 'grouped' ʹ��ͬ״̬������ʾ

% ���ͼ���ǩ
xlabel('Test Run');
ylabel('Accuracy (%)');
title('Accuracy Comparison Across Different States');

% ���ͼ��
legend('State 1', 'State 2', 'State 3');

% ���������
grid on;
% ������������������������������������������������������������������״ͼ���������������������������������������������������������������� %



end
total_tests = numTests * 3;  % �ܵĲ��Դ���
success_rate = (successful_status / total_tests) * 100;  % ����ɹ���

% ����ܲ��Դ������ɹ�Ԥ������Լ��ɹ���
fprintf('�� %d �β����У����ɹ�Ԥ�� %d �Ρ�\n', total_tests, successful_status);
fprintf('���гɹ���Ϊ %.2f%%��\n', success_rate);  % ����ɹ���
