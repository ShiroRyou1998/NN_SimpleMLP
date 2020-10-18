%%%%% Simple MLP model v0.1.5 %%%%%
% a NN of MLP program,test data is housing of poston
% used to conduct function regression
% creadit by shiro_ryou in 2020/10/13
% version Info.
% v.0.1.0  very basic MLP,can do BP propagation task 20/10/13
% v.0.1.5  few bugs correted,can randomly regress Function 20/10/15

clc
clear
tic

%%%% initialize network parameter %%%%
netLayerSize=[13 10 1];
studyRate=0.0001;
momentRate=0.5;
selectRate=0.2;
actOption='tanh_LeCun';

%%%% load data %%%%
load('housingposton.mat');%dataVar should be named by 'originData',M0*N

[numVar,sizeData]=size(originData);
inputData=originData(1:numVar-1,:);
expectRes=originData(numVar,:);%single output

%%%% initialize NN weight %%%%
netLayerNum=length(netLayerSize);

weightNum=0;
weightLayerNum=[0];
for i=1:netLayerNum-1
    weightNum=weightNum+netLayerSize(i)*netLayerSize(i+1);
    weightLayerNum=[weightLayerNum weightNum];
end

weightTemp=sqrt(1/weightNum)*(rand(1,weightNum)-0.5)/0.288;% rand return an array with E=0.5&D=0.288?
weightSpace=cell(1,netLayerNum-1);
for i=1:netLayerNum-1
    rowTemp=weightTemp((weightLayerNum(i)+1):weightLayerNum(i+1));
    weightSpace{i}=reshape(rowTemp,[netLayerSize(i),netLayerSize(i+1)]);
end

ZweiOriginSpace=weightSpace;% for check use

%%%% Neural Network Startup %%%%
iterationTimes=1;
MSE=[];TMSE=[];
predictMSE=114514;

while~(predictMSE<15)&&(iterationTimes<=1000)
    
    %%% select sample set for estimation %%%
    testSampleNum=floor(sizeData*selectRate);
    estmtSampleNum=sizeData-testSampleNum;
    testSampleIndex=randperm(sizeData,testSampleNum);
    estmtSampleIndex=setdiff([1:sizeData],testSampleIndex);
    
    %%% standardization %%%
    inputXData=inputData(:,estmtSampleIndex);
    [inputXData,PSX]=mapminmax(inputXData);
    [outputY,PSY]=mapminmax(expectRes(estmtSampleIndex));
    
    %%% main training %%%
    delayWeightDelta=cell(1,estmtSampleNum);
    
    for i=1:estmtSampleNum
        inputX=inputXData(:,i);
        
        layerY=cell(1,netLayerNum);
        layerY{1}=inputX;
        
        %%% forward propagation %%%
        for j=1:netLayerNum-1
            weightMat=weightSpace{j};
            [layerOutput,inputX]=feedForwardProp(inputX,weightMat,actOption);
            layerY{j+1}=layerOutput;
        end
        
        errorX=outputY(i)-layerOutput;
        
        %%% error feedback propagation %%%
        partDelta=errorX;
        weightDelta=cell(1,netLayerNum-1);
        
        for j=1:netLayerNum-1
            
            tupniYj=layerY{netLayerNum-j+1};
            tupniYi=layerY{netLayerNum-j};
            
            if j==1
                [weightTemp,partDelta]=feedBackProp(tupniYj,tupniYi,[1],partDelta,actOption);
            else
                [weightTemp,partDelta]=feedBackProp(tupniYj,tupniYi,weightMat,partDelta,actOption);
            end
            
            if i==1
                weightDelta{netLayerNum-j}=studyRate*weightTemp;
            else
                delayWeight=delayWeightDelta{i-1};
                weightDelta{netLayerNum-j}=studyRate*weightTemp+momentRate*delayWeight{netLayerNum-j};
            end
            
            weightMat=weightSpace{netLayerNum-j}; 
        end
             
        %%% weight modification %%%
        
        delayWeightDelta{i}=weightDelta;
        for j=1:netLayerNum-1
            weightSpace{j}=weightSpace{j}+weightDelta{j};
        end
        
    end
    
    %%%% NN TrianSet Caculation %%%%
    trainEndTrainY=zeros(1,estmtSampleNum);
    
    for i=1:estmtSampleNum
        inputX=inputXData(:,i);
        
        %%% forward propagation %%%
        for j=1:netLayerNum-1
            weightMat=weightSpace{j};
            [layerOutput,inputX]=feedForwardProp(inputX,weightMat,actOption);
        end
        trainEndTrainY(i)=layerOutput;
        
    end
    
    reverseTrianY=mapminmax('reverse',trainEndTrainY,PSY);
    predictMSE=sum((reverseTrianY-expectRes(estmtSampleIndex)).^2)/estmtSampleNum;
    
    TMSE=[TMSE predictMSE];
    
    %%%% NN TestSet Caculation %%%%
    trainEndY=zeros(1,testSampleNum);
    
    %%% standardization %%%
    [inputXData,~]=mapminmax(inputData(:,testSampleIndex));
    
    for i=1:testSampleNum
        inputX=inputXData(:,i);
        
        %%% forward propagation %%%
        for j=1:netLayerNum-1
            weightMat=weightSpace{j};
            [layerOutput,inputX]=feedForwardProp(inputX,weightMat,actOption);
        end
        trainEndY(i)=layerOutput;
        
    end
    
    reverseY=mapminmax('reverse',trainEndY,PSY);
    predictY=[expectRes(testSampleIndex);reverseY];
    predictMSE=sum((reverseY-expectRes(testSampleIndex)).^2)/testSampleNum;
    
    iterationTimes=iterationTimes+1;
    MSE=[MSE predictMSE];
    
    
end

%%%% plot result for comparasion %%%%

figure(1)
plot([1:testSampleNum],reverseY,'r-*')
hold on
plot([1:testSampleNum],expectRes(testSampleIndex),'b-o')
grid on
%}
figure(2)
plot([1:length(MSE)],MSE,'r','linewidth',0.7)
hold on
plot([1:length(TMSE)],TMSE,'b','linewidth',0.4)
grid on

toc