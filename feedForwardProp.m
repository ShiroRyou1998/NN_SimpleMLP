function [layerOutput,inputX]=feedForwardProp(inputX,weightMat,actOption)

%this function is to conduct caculation forward
%input: input variables(m1*1) weight of layer(m1*N) activation function mode
%output: v=sum(w*x), y=phi(v)

induceV=weightMat'*inputX;

switch actOption
    case 'tanh_LeCun'
        layerOutput=1.7159*tanh(2*induceV/3);
    case 'logistic'
        layerOutput=1/(1+exp(-induceV));
    otherwise
        layerOutput=tanh(induceV);
end

inputX=layerOutput;