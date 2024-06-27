classdef MOEADFFO < ALGORITHM
% <multi/many> <real/integer/label/binary/permutation>
% Multiobjective evolutionary algorithm based on decomposition
% type --- 1 --- The type of aggregation function

%------------------------------- Reference --------------------------------
% Q. Zhang and H. Li, MOEA/D: A multiobjective evolutionary algorithm based
% on decomposition, IEEE Transactions on Evolutionary Computation, 2007,
% 11(6): 712-731.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            type = Algorithm.ParameterSet(1);

            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            T = ceil(Problem.N/10);

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population = Problem.Initialization();
            Z = min(Population.objs,[],1);

%飞狐的基本参数计算与调整
N = Problem.N;
M = Problem.M;
D = Problem.D;
Xmin = Problem.lower;
Xmax = Problem.upper;
lamdaMat = W;
%X替换为---Population Population.decs Population.objs Population.cons
%计算更新方法 obj.CalDec(newDecs) obj.CalObj(newDecs) obj.CalObj(newDecs)


BestSolution=ones(N,1);
WorstSolution=ones(N,1);
Totalcost =zeros(N,1);
SurvivalList = zeros(N,1);

% 初始化 SurvivalList
SurvivalList = (1:N).';

% 计算 Totalcost
Totalcost = sum(Population(:).objs .* W, 2);


for i=1:N
% 计算当前个体群中每个个体的总成本
current_costs = Totalcost(B(i,:));

% 找到最佳解的索引
[~, best_idx] = min(current_costs);
BestSolution(i) = best_idx;

% 找到最差解的索引
[~, worst_idx] = max(current_costs);
WorstSolution(i) = worst_idx;
end



            %% Optimization
            while Algorithm.NotTerminated(Population)
                % For each solution
                for i = 1 : Problem.N

                    % 赋值方法===============================================================================================================
                    % y=Population(i).decs;
                    % x=Population(B(i,BestSolution(i))).decs;
                    % q=y+x;
                    % q=Problem.Evaluation(q);
                    % Population(i)=q;
                    % a=Population(i).dec(i);
                    %主要迭代部分

if norm(Totalcost(i)-Totalcost(B(i,BestSolution(i))))>0.14*norm(Totalcost(B(i,BestSolution(i)))-Totalcost(B(i,WorstSolution(i))))
if norm(Totalcost(i)-Totalcost(B(i,BestSolution(i))))>0.15*norm(Totalcost(B(i,BestSolution(i)))-Totalcost(B(i,WorstSolution(i))))
%过远距离的飞狐算子以moead方式操作
% Choose the parents
P = B(i,randperm(size(B,2)));
% Generate an offspring
y = OperatorGAhalf(Problem,Population(P(1:2)));
%过远距离飞狐策略结束
else
%远距离飞狐策略开始
y=Population(i).dec+(Population(B(i,BestSolution(i))).dec-Population(i).dec).*unifrnd(0,1,1,D)*0.5;%------------alpha暂时用0.5代替，delata(1)暂用0.3代替，ProblemSize用N暂代
y=Problem.Evaluation(y);
end
%远距离飞狐策略结束

%已经更改
else

%对于距离过近的飞狐执行操作
A=randperm(T); A(A==i)=[]; a=A(1); b=A(2);
stepsize=(Population(B(i,BestSolution(i))).dec-Population(i).dec).*unifrnd(0,1,1,D)+(Population(B(i,a)).dec-Population(B(i,b)).dec).*unifrnd(0,1,1,D);
y=zeros(size(Population(a).dec));
j0=randi([1 numel(Population(a).dec)]);

for j=1:numel(Population(a).dec)
if j==j0 || rand>=0.5 %------------pa暂时用0.5代替
y(j)=Population(i).dec(j)+stepsize(j);
else
y(j)=Population(i).dec(j);
end
end
y=Problem.Evaluation(y);


end

%对于距离过近的飞狐执行操作结束

%主要迭代部分结束
%主要迭代部分结束，第一次进行评估
Z = min(Z,y.obj);
                    % Update the neighbours
                    switch type
                        case 1
                            % PBI approach
                            normW   = sqrt(sum(W(B(i,:),:).^2,2));
                            normP   = sqrt(sum((Population(B(i,:)).objs-repmat(Z,T,1)).^2,2));
                            normO   = sqrt(sum((y.obj-Z).^2,2));
                            CosineP = sum((Population(B(i,:)).objs-repmat(Z,T,1)).*W(B(i,:),:),2)./normW./normP;
                            CosineO = sum(repmat(y.obj-Z,T,1).*W(B(i,:),:),2)./normW./normO;
                            g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                            g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2);
                        case 2
                            % Tchebycheff approach
                            g_old = max(abs(Population(B(i,:)).objs-repmat(Z,T,1)).*W(B(i,:),:),[],2);
                            g_new = max(repmat(abs(y.obj-Z),T,1).*W(B(i,:),:),[],2);
                        case 3
                            % Tchebycheff approach with normalization
                            Zmax  = max(Population.objs,[],1);
                            g_old = max(abs(Population(B(i,:)).objs-repmat(Z,T,1))./repmat(Zmax-Z,T,1).*W(B(i,:),:),[],2);
                            g_new = max(repmat(abs(y.obj-Z)./(Zmax-Z),T,1).*W(B(i,:),:),[],2);
                        case 4
                            % Modified Tchebycheff approach
                            g_old = max(abs(Population(B(i,:)).objs-repmat(Z,T,1))./W(B(i,:),:),[],2);
                            g_new = max(repmat(abs(y.obj-Z),T,1)./W(B(i,:),:),[],2);
                    end
                    Population(B(i,g_old>=g_new)) = y;
                    


% 获取符合条件的个体的索引
indices = B(i,g_old>=g_new);

% 循环处理每个个体
for idx = 1:length(indices)
    % 获取当前个体的索引
    current_idx = indices(idx);
    
    % 计算当前个体的总成本
    total_cost = sum(Population(current_idx).objs .* W(current_idx, :));
    
    % 更新 Totalcost
    Totalcost(current_idx) = total_cost;
end





% 计算当前个体群中每个个体的总成本
current_costs = Totalcost(B(i,:));

% 找到最佳解的索引
[~, best_idx] = min(current_costs);
BestSolution(i) = best_idx;

% 找到最差解的索引
[~, worst_idx] = max(current_costs);
WorstSolution(i) = worst_idx;




%第一次评估结束
%主要迭代部分结束


%窒息部分开始
for j=1:M
pBestFF=(find([Totalcost] == Totalcost(B(i,BestSolution(i))) ==1));
nBestFF=size(pBestFF,2);
pDeath=(nBestFF-1)/N;


for i=1:2:nBestFF
if rand<pDeath
j = 1:nPop;
j(pBestFF) = [];

%判断基数偶数
if mod(nBestFF,2)==1 && i==nBestFF
Population(pBestFF(i)).dec=ReplaceWithSurvivalList(SurvivalList,D,Population.dec);
Population(pBestFF(i))= Problem.Evaluation(Population(pBestFF,2).dec);
SurvivalList=UpdateSurvivalList(SurvivalList,SurvList,pBestFF(i),Population.dec,Totalcost);


% Improvement
% 第二次进行评估
Z = min(Z,Population(pBestFF(i)).obj);
                    % Update the neighbours
                    switch type
                        case 1
                            % PBI approach
                            normW   = sqrt(sum(W(B(pBestFF(i),:),:).^2,2));
                            normP   = sqrt(sum((Population(B(pBestFF(i),:)).objs-repmat(Z,T,1)).^2,2));
                            normO   = sqrt(sum((Population(pBestFF(i)).obj-Z).^2,2));
                            CosineP = sum((Population(B(pBestFF(i),:)).objs-repmat(Z,T,1)).*W(B(pBestFF(i),:),:),2)./normW./normP;
                            CosineO = sum(repmat(Population(pBestFF(i)).obj-Z,T,1).*W(B(pBestFF(i),:),:),2)./normW./normO;
                            g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                            g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2);
                        case 2
                            % Tchebycheff approach
                            g_old = max(abs(Population(B(pBestFF(i),:)).objs-repmat(Z,T,1)).*W(B(pBestFF(i),:),:),[],2);
                            g_new = max(repmat(abs(Population(pBestFF(i)).obj-Z),T,1).*W(B(pBestFF(i),:),:),[],2);
                        case 3
                            % Tchebycheff approach with normalization
                            Zmax  = max(Population.objs,[],1);
                            g_old = max(abs(Population(B(pBestFF(i),:)).objs-repmat(Z,T,1))./repmat(Zmax-Z,T,1).*W(B(pBestFF(i),:),:),[],2);
                            g_new = max(repmat(abs(Population(pBestFF(i)).obj-Z)./(Zmax-Z),T,1).*W(B(pBestFF(i),:),:),[],2);
                        case 4
                            % Modified Tchebycheff approach
                            g_old = max(abs(Population(B(pBestFF(i),:)).objs-repmat(Z,T,1))./W(B(pBestFF(i),:),:),[],2);
                            g_new = max(repmat(abs(Population(pBestFF(i)).obj-Z),T,1)./W(B(pBestFF(i),:),:),[],2);
                    end
                    Population(B(pBestFF(i),g_old>=g_new)) = Population(pBestFF(i));





% 获取符合条件的个体的索引
indices = B(i,g_old>=g_new);

% 循环处理每个个体
for idx = 1:length(indices)
    % 获取当前个体的索引
    current_idx = indices(idx);
    
    % 计算当前个体的总成本
    total_cost = sum(Population(current_idx).objs .* W(current_idx, :));
    
    % 更新 Totalcost
    Totalcost(current_idx) = total_cost;
end

% 计算当前个体群中每个个体的总成本
current_costs = Totalcost(B(i,:));

% 找到最佳解的索引
[~, best_idx] = min(current_costs);
BestSolution(i) = best_idx;

% 找到最差解的索引
[~, worst_idx] = max(current_costs);
WorstSolution(i) = worst_idx;

%第二次评估结束


%另一种情况进行窒息操作
else


%else开始
if length(pBestFF)>1
parent1=randi(N);
parent2=randi(N);
if rand<0.5 && Totalcost(parent1)~=Totalcost(parent2)
[Population(pBestFF(i)).dec, Population(pBestFF(i+1)).dec]=Crossover(Population(parent1).dec,Population(parent2).dec,Xmin,Xmax);
else
Population(pBestFF(i)).dec=ReplaceWithSurvivalList(SurvivalList,D,Population.dec);
Population(pBestFF(i+1)).dec=ReplaceWithSurvivalList(SurvivalList,D,Population.dec);
end
Population(pBestFF(i)) = Problem.Evaluation(Population(pBestFF(i)).dec);
SurvivalList=UpdateSurvivalList(SurvivalList,SurvList,pBestFF(i),Population.dec,Totalcost);

Population(pBestFF(i+1)) = Problem.Evaluation(Population(pBestFF(i+1)).dec);
SurvivalList=UpdateSurvivalList(SurvivalList,SurvList,pBestFF(i+1),Population.dec,Totalcost);



%第二次评估
Z = min(Z,Population(pBestFF(i)).obj);
                    % Update the neighbours
                    switch type
                        case 1
                            % PBI approach
                            normW   = sqrt(sum(W(B(pBestFF(i),:),:).^2,2));
                            normP   = sqrt(sum((Population(B(pBestFF(i),:)).objs-repmat(Z,T,1)).^2,2));
                            normO   = sqrt(sum((Population(pBestFF(i)).obj-Z).^2,2));
                            CosineP = sum((Population(B(pBestFF(i),:)).objs-repmat(Z,T,1)).*W(B(pBestFF(i),:),:),2)./normW./normP;
                            CosineO = sum(repmat(Population(pBestFF(i)).obj-Z,T,1).*W(B(pBestFF(i),:),:),2)./normW./normO;
                            g_old   = normP.*CosineP + 5*normP.*sqrt(1-CosineP.^2);
                            g_new   = normO.*CosineO + 5*normO.*sqrt(1-CosineO.^2);
                        case 2
                            % Tchebycheff approach
                            g_old = max(abs(Population(B(pBestFF(i),:)).objs-repmat(Z,T,1)).*W(B(pBestFF(i),:),:),[],2);
                            g_new = max(repmat(abs(Population(pBestFF(i)).obj-Z),T,1).*W(B(pBestFF(i),:),:),[],2);
                        case 3
                            % Tchebycheff approach with normalization
                            Zmax  = max(Population.objs,[],1);
                            g_old = max(abs(Population(B(pBestFF(i),:)).objs-repmat(Z,T,1))./repmat(Zmax-Z,T,1).*W(B(pBestFF(i),:),:),[],2);
                            g_new = max(repmat(abs(Population(pBestFF(i)).obj-Z)./(Zmax-Z),T,1).*W(B(pBestFF(i),:),:),[],2);
                        case 4
                            % Modified Tchebycheff approach
                            g_old = max(abs(Population(B(pBestFF(i),:)).objs-repmat(Z,T,1))./W(B(pBestFF(i),:),:),[],2);
                            g_new = max(repmat(abs(Population(pBestFF(i)).obj-Z),T,1)./W(B(pBestFF(i),:),:),[],2);
                    end
                    Population(B(pBestFF(i),g_old>=g_new)) = Population(pBestFF(i));





% 获取符合条件的个体的索引
indices = B(i,g_old>=g_new);

% 循环处理每个个体
for idx = 1:length(indices)
    % 获取当前个体的索引
    current_idx = indices(idx);
    
    % 计算当前个体的总成本
    total_cost = sum(Population(current_idx).objs .* W(current_idx, :));
    
    % 更新 Totalcost
    Totalcost(current_idx) = total_cost;
end

    
for j=1:length(B(i,:))
if Totalcost(B(i,j))<Totalcost(B(i,BestSolution(i)))
BestSolution(i)=j;
end
if Totalcost(B(i,j))>Totalcost(B(i,WorstSolution(i)))
WorstSolution(i)=j;
end

end
%第二次评估结束
end
%else结束
end
end
end
end
%窒息部分结束




                   
                end
            end
        end
    end
end



function SurvivalList=UpdateSurvivalList(SurvivalList,SurvList,temp,X,Totalcost)
if Totalcost(temp)<Totalcost(SurvivalList(end))
SurvivalList=[temp;SurvivalList];
[~,ii]=unique((([costofSL(SurvivalList,Totalcost)])));
SurvivalList=SurvivalList(ii);
if size(SurvivalList,1)>SurvList
SurvivalList=SurvivalList(1:SurvList);
end
end
end

function ffox=ReplaceWithSurvivalList(SurvivalList,ProblemSize,X)
m=randi([2 size(SurvivalList,1)]);
h=randperm(size(SurvivalList,1),m);
ffox=zeros(1,ProblemSize);
for i=1:size(h,2)
ffox=ffox+X(SurvivalList(h(i)),:);
end
ffox=ffox/m;
end


%计算SurvivalList.cost
function cost= costofSL(temp,Totalcost)
cost = zeros(length(temp),1);
for i = 1:length(temp)
cost(i) = Totalcost(temp(i));
end
end

function [off1, off2]=Crossover(x1,x2,LowerBound,UpperBound)
extracros=0.0;
L=unifrnd(-extracros,1+extracros,size(x1));
off1=L.*x1+(1-L).*x2; off2=L.*x2+(1-L).*x1;
off1=max(off1,LowerBound); off1=min(off1,UpperBound);% Position Limits
off2=max(off2,LowerBound); off2=min(off2,UpperBound);
end





