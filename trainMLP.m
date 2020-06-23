%% *** code instruction **
%{
--- this code is used to train and evaluate the MLP models with HOG and LBP
features
--- the code is only for reference and cannot be run because the actual
data is not included, the final best models have been saved explicitly
%}
%% create training and testing set 
face_gallery=imageDatastore('~/Desktop/Computer Vision CW/face_only'...
  ,'IncludeSubfolders',true,'Labelsource','foldernames');
[training,test]=splitEachLabel(face_gallery,0.7);

%% extract HOG features
trainingFeatures_HOG=[];
testingFeatures_HOG=[];

for i=1:size(training.Files,1)
    img=readimage(training,i);
    img=imresize(img,[227 227]);

    trainingFeatures_HOG(i,:)=extractHOGFeatures(img);
end
for i=1:size(test.Files,1)
    img=readimage(test,i);
    img=imresize(img,[227 227]);

    testingFeatures_HOG(i,:)=extractHOGFeatures(img);
end
%% extract LBP features
for i=1:size(training.Files,1)
    img=rgb2gray(readimage(training,i));
    trainingFeatures_LBP(i,:)=extractLBPFeatures(img);
end
for i=1:size(test.Files,1)
    img=rgb2gray(readimage(test,i));
    testingFeatures_LBP(i,:)=extractLBPFeatures(img);
end
%% MLP with HOG
% note: the following codes with commands are the setting of
% hyperparameters that used to train, hyperparameters that without commands
% are the best hyperparameter setting

x = trainingFeatures_HOG';

%% Getting target labels for training set
t = grp2idx(training.Labels);
t_train = zeros(size(t,1),69); % 69 is the number of classes 
for i= 1:69
    rows = t == i;
    t_train(rows,i) = 1;
end 

%% Getting target labels for test set
t2 = grp2idx(test.Labels);

t_test = zeros(size(t2,1),69); % 69 is the number of classes 

for i= 1:69
    rows = t2 == i;
    t_test(rows,i) = 1;
end 

result_MLP=["Number of Hidden Layers", "Activation Function", "Layer Size", "Tr_Accuracy"...
    ,"Te_Accuracy"];
trainFcn = 'trainscg';
hiddenlayers=[1];
%hiddenlayers=[1 2 3];
layersize=[40]
%layersize=[5 10 15 20 25 30 35 40];
activationFcn={'tansig'};
%activationFcn={'logsig' 'tansig' 'purelin'};
k=2
for n=1:size(hiddenlayers,2) % loop over number of hidden layers
    numhidlayers=hiddenlayers(n);
    for f=1:size(activationFcn,2) % loop over activation function
        for s=1:size(layersize,2) % loop over size of layer
            net = patternnet(ones(1,hiddenlayers(n))*layersize(s),trainFcn); % initializing a MLP
            net.numLayers=numhidlayers+1;
            for l=1:numhidlayers % assign activation function to each layer
                net.layers{l}.transferFcn=activationFcn{f};
            end
            net.input.processFcns = {'removeconstantrows','mapminmax'};
            net.divideFcn = 'dividerand';  
            net.divideMode = 'sample';  
            net.divideParam. trainRatio= 0.8;
            net.divideParam.valRatio = 0.1;
            net.divideParam. testRatio= 0.1;
            net.performFcn = 'crossentropy';  % Choose a Performance Function
            [net,tr] = train(net,x,t_train'); % train a MLP
            MLP_HOG_net = net;
            

            % Evaluate training result
            tr_pred = net(x);
            tr_pred_ind=vec2ind(tr_pred);
            te_pred=net(testingFeatures_HOG');
            te_pred_ind=vec2ind(te_pred);
            tr_ind=vec2ind(t_train');
            te_ind=vec2ind(t_test');
            percentErrors_train = sum(tr_pred_ind ~= tr_ind)/numel(tr_pred_ind);
            percentErrors_test = sum(te_pred_ind ~= te_ind)/numel(te_pred_ind);
            result_MLP(k,1)=hiddenlayers(n);
            result_MLP(k,2)=activationFcn{f};
            result_MLP(k,3)=layersize(s);
            result_MLP(k,4)=1-percentErrors_train;
            result_MLP(k,5)=1-percentErrors_test;
            k=k+1
        end
    end
end

% confusion matrix 
plotconfusion(te_ind, te_pred_ind)
title('Confusion Matrix from MLP model - HOG)');

%already saved, so we comment out
% save MLP_HOG_net
%             whos net MLP_HOG_net
%             clear net MLP_HOG_net
%% MLP with LBP
% note: the following codes with commands are the setting of
% hyperparameters that used to train, hyperparameters that without commands
% are the best hyperparameter setting

x = trainingFeatures_LBP';


result_MLP=["Number of Hidden Layers", "Activation Function", "Layer Size", "Tr_Accuracy"...
    ,"Te_Accuracy"];
trainFcn = 'trainscg';
hiddenlayers=[1];
%hiddenlayers=[1 2 3];
layersize=[25]
%layersize=[5 10 15 20 25 30 35 40];
activationFcn={'tansig'};
%activationFcn={'logsig' 'tansig' 'purelin'};
k=2
for n=1:size(hiddenlayers,2) % loop over number of hidden layers
    numhidlayers=hiddenlayers(n);
    for f=1:size(activationFcn,2) % loop over activation function
        for s=1:size(layersize,2) % loop over size of layer
            net = patternnet(ones(1,hiddenlayers(n))*layersize(s),trainFcn); % initializing a MLP
            net.numLayers=numhidlayers+1;
            for l=1:numhidlayers % assign activation function to each layer
                net.layers{l}.transferFcn=activationFcn{f};
            end
            net.input.processFcns = {'removeconstantrows','mapminmax'};
            net.divideFcn = 'dividerand'; 
            net.divideMode = 'sample';  
            net.divideParam. trainRatio= 0.8;
            net.divideParam.valRatio = 0.1;
            net.divideParam. testRatio= 0.1;
            net.performFcn = 'crossentropy';  % Choose a Performance Function
            [net,tr] = train(net,x,t_train'); % train a MLP
            MLP_LBP_net = net;
            
            % Evaluate training result
            tr_pred = net(x);
            tr_pred_ind=vec2ind(tr_pred);
            te_pred=net(testingFeatures_LBP');
            te_pred_ind=vec2ind(te_pred);
            tr_ind=vec2ind(t_train');
            te_ind=vec2ind(t_test');
            percentErrors_train = sum(tr_pred_ind ~= tr_ind)/numel(tr_pred_ind);
            percentErrors_test = sum(te_pred_ind ~= te_ind)/numel(te_pred_ind);
            result_MLP(k,1)=hiddenlayers(n);
            result_MLP(k,2)=activationFcn{f};
            result_MLP(k,3)=layersize(s);
            result_MLP(k,4)=1-percentErrors_train;
            result_MLP(k,5)=1-percentErrors_test;
            k=k+1
        end
    end
end

% confusion matrix 
plotconfusion(te_ind, te_pred_ind)
title('Confusion Matrix from MLP model - LBP)');


%already saved, so we comment out
% save MLP_LBP_net
% whos net MLP_LBPnet
% clear net MLP_LBP_net
