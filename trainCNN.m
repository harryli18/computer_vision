%% *** code instruction **
%{
--- this code is used to train and evaluate the CNN models with HOG and LBP
features
--- the code is only for reference and cannot be run because the actual
data is not included, the final best models have been saved explicitly
%}
%% create training and testing set 
face_gallery=imageDatastore('~/Desktop/Computer Vision CW/face_only'...
  ,'IncludeSubfolders',true,'Labelsource','foldernames');
[train,test]=splitEachLabel(face_gallery,0.7);

for i=1:size(train.Files,1)
    img=readimage(train,i);
    img=imresize(img,[527 527]);

end
for i=1:size(test.Files,1)
    img=readimage(test,i);
    img=imresize(img,[527 527]);
end
%% train CNN model
net = alexnet;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(train.Labels))
layers = [
layersTransfer
fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
softmaxLayer
classificationLayer];
miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(train.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
'MiniBatchSize',miniBatchSize,...
'MaxEpochs',4,...
'InitialLearnRate',1e-4,...
'Verbose',false,...
'Plots','training-progress',...
'ValidationData',test,...
'ValidationFrequency',numIterationsPerEpoch);
netTransfer = trainNetwork(train,layers,options);

CNN_alexnet = netTransfer;
predictedLabels_tr = classify(netTransfer,train);
tr_accuracy=1-sum(predictedLabels_tr~=train.Labels)/size(train.Labels,1)
predictedLabels_te = classify(netTransfer,test);
te_accuracy=1-sum(predictedLabels_te~=test.Labels)/size(test.Labels,1)

% save CNN_alexnet 
% clear net CNN_alexnet
