%% *** code instruction ***
%{
--- this function takes input of an image, feature type and classifier to
do a face recognition task

--- make sure the following files are in the same directory as this
funtion:
- 'SVM_SURF_Classifier.mat'
- 'SVM_HOG_Classifier.mat'
- 'CNN_alexnet.mat'
- 'MLP_LBP_net.mat'
- 'MLP_HOG_net.mat'

--- valid input arguments:
1) I, an input image, please make sure this image is in a straight position
(i.e. the person/object in the image is in straight position)
2) featureType argument takes one of the following: SURF, HOG, LBP and none
note: SURF and HOG are valid when classifierName is SVM, HOG and LBP are
valid when classifierName is MLP, none is valid with classifierName CNN
3) classifierName argument takes one of the following: SVM, MLP and CNN
note: SVM is valid when featureType is HOG and SURF, MLP is valid when
featureType is HOG and LBP, CNN is valid when featureType is none

--- output will be a N*3 matrix, where N is the number of faces detected
from the image, the first column is the person ID, second column is the
x-coordinate of centre of face, the last column is y-coordinate of centre of face
%}

function P = RecogniseFace(I,featureType,classifierName)

%showing initial image and the feature type and classifier that we are
%using 
imshow(I)
I = imread(I);
disp(['feature type:',featureType])
disp(['classifier:',classifierName])
    % initialize a face detector
    facedector=vision.CascadeObjectDetector();
    
    %initialise useful variables 
    P=["Person ID" "X" "Y"]; % define our output, the 3-D vector.
    k=2;
    r=1;
    roi=[]; %define region of interest 
    % detect face from the image and extract it
    
    
    box=step(facedector,I);
    for i=1:size(box,1)
        if box(i,3)>50 & box(i,4)>50
            roi(r,:)=box(i,:);
            r=r+1;
        end
    end
    % if no face has been detected, an empty matrix will be returned
    if length(roi)<1
        P=[];
        disp('no face has been detected')
    % perform the face recognition according to the selected feature type
    % and classifier
    else
        if strcmpi(featureType,'SURF') & strcmpi(classifierName,'CNN')
            disp('Classifier CNN does not have feature type option, set feature type to none when using CNN')
        end
        if strcmpi(featureType,'HOG') & strcmpi(classifierName,'CNN')
            disp('Classifier CNN does not have feature type option, set feature type to none when using CNN')
        end
        if strcmpi(featureType,'SURF') & strcmpi(classifierName,'MLP')
            disp('feature type for MLP is either HOG or LBP')
        end
        if strcmpi(featureType,'LBP') & strcmpi(classifierName,'SVM')
            disp('feature type for SVM is either HOG or SURF')
        end
        if strcmpi(featureType,'SURF') & strcmpi(classifierName,'SVM')
            load 'SVM_SURF_Classifier.mat'
            for i=1:size(roi,1)
                face=I(roi(i,2):roi(i,2)+roi(i,4),roi(i,1):roi(i,1)+roi(i,3),:);
                face=imresize(face,[227 227]); % need to resize to match training data to keep accuracy
                label=SVM_SURF_Classifier.Labels(predict(SVM_SURF_Classifier,face));
                I=insertObjectAnnotation(I,'rectangle',roi(i,:),char(label));
                imshow(I);
                hold on
                P(k,1)=char(label);
                P(k,2)=roi(i,1)+0.5*roi(i,3);
                P(k,3)=roi(i,2)+0.5*roi(i,4);
                k=k+1;
            end
        end
        if strcmpi(featureType,'HOG') & strcmpi(classifierName,'SVM')
            load 'SVM_HOG_Classifier.mat'
            for i=1:size(roi,1)
                face=I(roi(i,2):roi(i,2)+roi(i,4),roi(i,1):roi(i,1)+roi(i,3),:);
                face=imresize(face,[227 227]); % need to resize to match training data to keep accuracy
                face_HOG_features=extractHOGFeatures(face);
                size(face_HOG_features)
                label=predict(SVM_HOG_Classifier,face_HOG_features);
                I=insertObjectAnnotation(I,'rectangle',roi(i,:),char(label));
                imshow(I);
                hold on;
                P(k,1)=char(label);
                P(k,2)=roi(i,1)+0.5*roi(i,3);
                P(k,3)=roi(i,2)+0.5*roi(i,4);
                k=k+1;
            end
        end
        if strcmpi(featureType,'none') & strcmpi(classifierName,'CNN')
            load 'CNN_alexnet.mat'
            for i=1:size(roi,1)
                face=I(roi(i,2):roi(i,2)+roi(i,4),roi(i,1):roi(i,1)+roi(i,3),:);
                face=imresize(face,[227 227]); % need to resize to match training data to keep accuracy
                label=classify(CNN_alexnet,face);
                I=insertObjectAnnotation(I,'rectangle',roi(i,:),char(label));
                imshow(I);
                hold on
                P(k,1)=char(label);
                P(k,2)=roi(i,1)+0.5*roi(i,3);
                P(k,3)=roi(i,2)+0.5*roi(i,4);
                k=k+1;
            end
        end
        if strcmpi(featureType,'LBP') & strcmpi(classifierName,'MLP')
            load 'MLP_LBP_net.mat'
%             load 'class_label_index.mat'
            for i=1:size(roi,1)
                face=I(roi(i,2):roi(i,2)+roi(i,4),roi(i,1):roi(i,1)+roi(i,3),:);
                face=rgb2gray(face);
                face=imresize(face,[227 227]); % need to resize to match training data to keep accuracy
                face_LBP_features=extractLBPFeatures(face);
                ind=vec2ind(MLP_LBP_net(face_LBP_features'));
                label=ind;
                I=insertObjectAnnotation(I,'rectangle',roi(i,:),label);
                imshow(I);
                hold on
                P(k,1)=label;
                P(k,2)=roi(i,1)+0.5*roi(i,3);
                P(k,3)=roi(i,2)+0.5*roi(i,4);
                k=k+1;
            end
        end
        if strcmpi(featureType,'HOG') & strcmpi(classifierName,'MLP')
            load 'MLP_HOG_net.mat'
%             load 'class_label_index.mat'
            for i=1:size(roi,1)
                face=I(roi(i,2):roi(i,2)+roi(i,4),roi(i,1):roi(i,1)+roi(i,3),:);
                face=imresize(face,[227 227]); % need to resize to match training data to keep accuracy
                face_HOG_features=extractHOGFeatures(face);
                ind=vec2ind(net(face_HOG_features'));
                label = ind;
%                label=class_label_index(ind);
                I=insertObjectAnnotation(I,'rectangle',roi(i,:),label);
                imshow(I);
                hold on
                P(k,1)=label;
                P(k,2)=roi(i,1)+0.5*roi(i,3);
                P(k,3)=roi(i,2)+0.5*roi(i,4);
                k=k+1;
            end
        end
    end
