function [weightTemp,partDelta]=feedBackProp(tupniYj,tupniYi,weightMat,partDelta,actOption)

% this function caculate weight delta and do the partdelta recursion
% input: tupniYj(ori-output),tupniYi(ori-input),weightMat(ji), partDelta(k)
% output:weightdelta(ji),partDelta(j)

ERRORX=weightMat*partDelta;
switch actOption
    case 'tanh_LeCun'
        a=1.7159;b=2/3;
        partDelta=(b/a*(a-tupniYj).*(a+tupniYj)).*ERRORX;
    otherwise
        partDelta=tupniYj.*(1-tupniYj).*ERRORX;
end

weightTemp=tupniYi*partDelta';

