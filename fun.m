%% ##########  给定优化的目标函数
function f=fun(x)
global L
global M
N=3;
%global AllData
global TrainData

T = length( TrainData );


for k=1:L
    for n=1:N
        beta1(k,n)=x((k-1)*N+n);
    end
end
for k = 1:L
    RuleWT(k) = x(L*N +k);
end
for k = 1:M
    AttributeWT(k) = x(L*N+L+k);
end

for n = 1:T
    %% 输入信息的转换为匹配度
    l1 = TrainData (n,1);
    l2 = TrainData (n,2);
    y1=[1.508063e+01    3.105238e-02    3.648781e-03    1.508345e-04    1.025396e-14];    % y1参考值    [3 2 3 1 1]CD
    y2=[1.082179e+01    4.968245e-02    8.106164e-03    3.992412e-03    9.233760e-13];    % y2参考值    [3 2 3 1 1]CA
    T1=length(y1);
    T2=length(y2);
    In=zeros(L,M);
    for i=1:T1-1
        for j=1:T2-1
            if l1<=y1(i) & l1>y1(i+1)   %%对于两个输出均为上升趋势的情况
                if l2<=y2(j) & l2>y2(j+1)
                    a2=(y1(i)-l1)/(y1(i)-y1(i+1)); %对左端点的置信度
                    a1=(l1-y1(i+1))/(y1(i)-y1(i+1)); %对右端点的置信度
                    b2=(y2(j)-l2)/(y2(j)-y2(j+1)); %对左端点的置信度
                    b1=(l2-y2(j+1))/(y2(j)-y2(j+1)); %对右端点的置信度
                    for k=1:T1
                        In((k-1)*T2+j,2)=b1;
                        In((k-1)*T2+j+1,2)=b2;
                    end
                    In((i-1)*T2+1:i*T2,1)=a1;
                    In(i*T2+1:(i+1)*T2,1)=a2;
                end
            end
        end
    end
    InputD(: , : , n ) = In ;
    
    for  k = 1:L
        weight(k) = 1;
        for m = 1:M
            weight(k) = weight(k) *  InputD (k,m,n);
        end
        if weight(k) == 0
            AM(k) = 0;
        else
%               AM(k) =  ( RuleWT(k) * ( InputD(k,1,n) ) ^AttributeWT(1) ) * ( RuleWT(k) * ( InputD(k,2,n) )^AttributeWT(2) ) ;
            AM(k) =  RuleWT(k) * (( InputD(k,1,n) ) ^AttributeWT(1) * ( InputD(k,2,n) )^AttributeWT(2) ) ;
        end
    end
    AU = sum( AM );
    for k =1:L
        ActivationW(n,k) = (AM(k)/ AU );      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%calculate激活权重
    end
    Doutput = [1   2   3];
    Sum1 = sum(  beta1' );
    for j = 1:N
        temp1(j) = 1;
        for k = 1:L
            Belief1 (k,j) = ActivationW(n,k) * beta1 (k,j) +1 - ActivationW(n,k) * Sum1(k);      %%%%%%%%系数的前半部分
            temp1(j) = temp1(j) * Belief1(k,j);
        end
    end
    temp2 = sum (temp1);
    temp3 = 1;
    for k = 1:L
        Belief2(k) = 1 - ActivationW(n,k)* Sum1(k);
        temp3 = temp3 * Belief2(k);                     %%%%%%%%%%%%系数的后半部分
    end
    Value = (temp2 - (N-1) * temp3)^-1;
    temp4 = 1;
    for k = 1:L
        temp4 = temp4 * (1 - ActivationW(n,k));
    end
    for j = 1:N
        BeliefOut(j) = ( Value * ( temp1(j) - temp3)) / ( 1 - Value * temp4);
    end
    y(n) = Doutput * (BeliefOut)';    %% BRB输出
    Mse_original(n) = ( y(n) - TrainData(n,3) )^2;
end
f = sum(Mse_original) /T;
end





