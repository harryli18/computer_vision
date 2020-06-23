

face_gallery=imageDatastore('~/Desktop/Computer Vision CW/FaceDataset v2'...
   ,'IncludeSubfolders',true,'Labelsource','foldernames');
labels = face_gallery.Labels;

% initialize a face detector
   facedector=vision.CascadeObjectDetector();
    P=["Person ID" "X" "Y"];
    k=2;
    r=1;
    roi=[];
    % detect face from the image and extract it
    
    for n = 1:size(face_gallery.Files, 1)
        img = readimage(face_gallery,n);
        box=step(facedector,img);
    for i=1:size(box,1)
        if box(i,3)>100 & box(i,4)>100 & box(i,3) < 1000 & box(i,4) < 1000
            roi(r,:)=box(i,:);
            r=r+1;
        end
    end
    for j=1:size(roi,1)
    face=img(roi(j,2):roi(j,2)+roi(j,4),roi(j,1):roi(j,1)+roi(j,3),:);
    face=imresize(face,[227,227]); % resize the image for accuracy
    
   
   
    end 
    imwrite(face,strcat('label',string(labels(n)),'id',num2str(n),'face.jpg'));
    r=1;
   roi=[];
    end
    
  
    