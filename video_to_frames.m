% this function extra frames from a video
% this function is designed to increase the size of the training and
% testing dataset for MLP, SVM and CNN. This can improve the accuracy of
% our model and reduce the over-fit.

function b = video_to_frames(filename);
V = VideoReader(filename);
for img = 1:V.NumberofFrames;
    filename=strcat('frame',num2str('img'),'.jpg');
    b = read(V, img);
    imwrite(b,filename);
    
 end
end 