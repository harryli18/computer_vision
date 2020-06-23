%% *** code instruction **

% create training and testing set 
face_gallery=imageDatastore('~/Desktop/Computer Vision CW/face_only'...
   ,'IncludeSubfolders',true,'Labelsource','foldernames');
[train,test]=splitEachLabel(face_gallery,0.7);

%% extract HOG features
trainingFeatures_HOG=[];
for i=1:size(train.Files,1)
    img=readimage(train,i);
    % different size of 
    img=imresize(img,[227 227]);
    trainingFeatures_HOG(i,:)=extractHOGFeatures(img);
end
testingFeatures_HOG=[];
for i=1:size(test.Files,1)

    img2=readimage(test,i);
    img2=imresize(img2,[227 227]);
    testingFeatures_HOG(i,:)=extractHOGFeatures(img2);
end
%% extract bag of feature with vocabulary size of 500 5000 and 50000(SURF)
% the result of bags have been saved for training\
bag_500 = bagOfFeatures(train);
bag_5000=bagOfFeatures(train,'VocabularySize',5000);
bag_50000=bagOfFeatures(train,'VocabularySize',50000);
bags={bag_500 bag_5000 bag_50000};
%% training models with HOG features
% note: the following codes with commands are the setting of
% hyperparameters that used to train, hyperparameters that without commands
% are the best hyperparameter setting
trainingLabel=train.Labels;
%kernel={'gaussian' 'polynomial' 'linear'};
kernel={'linear'};
%constraints=[0.1:0.5:5];
constraints=[0.6];
kernelorder=[5];
%kernelorder=[2 3 4 5];
res_SVM_HOG=["kernel", "kernelorder", "C", "accuracy"];
s=2
for k=1:size(kernel,2)
        for c=1:size(constraints,2)
            if strcmpi(char(kernel(k)),'polynomial')
                for order=1:size(kernelorder,2)
                    t=templateSVM('KernelFunction',char(kernel(k))...
                        ,'BoxConstraint',constraints(c)...
                        ,'PolynomialOrder',kernelorder(order));
                    SVM_HOG_Classifier = fitcecoc(trainingFeatures_HOG,trainingLabel...
                        ,'Learners',t); % this is only for the selected model
                     %SVM_HOG_Classifier = fitcecoc(trainingFeatures_HOG,trainingLabel...
                      %  ,'Learners',t,'CrossVal','on','KFold',5); this
                      %  will be run during training
                    label=predict(SVM_HOG_Classifier,testingFeatures_HOG);
                    Accuracy=1-sum(label ~= test.Labels)/numel(label);
                    %Accuracy=1-kfoldLoss(SVM_HOG_Classifier,'LossFun','ClassifError');
                    res_SVM_HOG(s,1)=kernel(k);
                    res_SVM_HOG(s,2)=kernelorder(order);
                    res_SVM_HOG(s,3)=constraints(c);
                    res_SVM_HOG(s,4)=Accuracy;
                    s=s+1
                end
            else
                t=templateSVM('KernelFunction',char(kernel(k))...
                    ,'BoxConstraint',constraints(c));
                SVM_HOG_Classifier = fitcecoc(trainingFeatures_HOG,trainingLabel...
                        ,'Learners',t); % this is only for the selected model
                    %SVM_HOG_Classifier = fitcecoc(trainingFeatures_HOG,trainingLabel...
                      %  ,'Learners',t,'CrossVal','on','KFold',5); this
                      %  will be run during training
                    %Accuracy=1-kfoldLoss(SVM_HOG_Classifier,'LossFun','ClassifError');
                    label=predict(SVM_HOG_Classifier,testingFeatures_HOG);
                    Accuracy=1-sum(label ~= test.Labels)/numel(label)
                    % plot confusion matrix 
                    plotconfusion(test.Labels, label)
                    title('Confusion Matrix from SVM model - HPG)');
                    res_SVM_HOG(s,1)=kernel(k);
                    res_SVM_HOG(s,2)='NA';
                    res_SVM_HOG(s,3)=constraints(c);
                    res_SVM_HOG(s,4)=Accuracy;
                    s=s+1
            end
        end
end
% save SVM_HOG_Classifier
% clear SVM_HOG_Classifier
%% train models with bag of features (SURF)
% note: the following codes with commands are the setting of
% hyperparameters that used to train, hyperparameters that without commands
% are the best hyperparameter setting

%kernel={'gaussian' 'polynomial' 'linear'};
kernel={'linear'};
constraints=[1.1];
%constraints=[0.1:0.5:5];
%kernelorder=[2 3 4 5];
bags={bag_50000};
%bags={bag_500 bag_5000 bag_50000};
SURFres=["vocabularysize", "kernel", "kernelorder", "C", "accuracy"];
s=2;
for b=1:size(bags,2)
    for k=1:size(kernel,2)
        for c=1:size(constraints,2)
            if strcmpi(char(kernel(k)),'polynomial')
                for order=1:size(kernelorder,2)
                    t=templateSVM('KernelFunction',char(kernel(k))...
                        ,'BoxConstraint',constraints(c)...
                        ,'PolynomialOrder',kernelorder(order));
                   SVM_SURF_Classifier = trainImageCategoryClassifier(train, bags{1,b}...
                        ,'LearnerOptions',t);
                    confMatrix = evaluate(SVM_SURF_Classifier, test); %accuracy in this model is evaluated on testing set
                    SURFres(s,1)=bags{1,b}.VocabularySize;
                    SURFres(s,2)=kernel(k);
                    SURFres(s,3)=kernelorder(order);
                    SURFres(s,4)=constraints(c);
                    SURFres(s,5)=mean(diag(confMatrix));
                    s=s+1;
                end
            else
                t=templateSVM('KernelFunction',char(kernel(k))...
                    ,'BoxConstraint',constraints(c));
                SVM_SURF_Classifier = trainImageCategoryClassifier(train, bags{1,b}...
                    ,'LearnerOptions',t);
                    
                    confMatrix = evaluate(SVM_SURF_Classifier, test); %accuracy in this model is evaluated on testing set
                    % plot confusion matrix 
                    plotconfusion(SVM_SURF_Classifier, test)
                    title('Confusion Matrix from SVM model - SURF)');
                    SURFres(s,1)=bags{1,b}.VocabularySize;
                    SURFres(s,2)=kernel(k);
                    SURFres(s,3)='NA';
                    SURFres(s,4)=constraints(c);
                    SURFres(s,5)=mean(diag(confMatrix));
                    s=s+1;
            end
        end
    end
end  

%already saved, hence we commented out
% save SVM_SURF_Classifier
% clear SVM_SURF_Classifier