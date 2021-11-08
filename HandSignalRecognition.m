net=alexnet;
net.Layers
inputSize = net.Layers(1).InputSize

imdsTrain = imageDatastore(fullfile('D:\','data'),...
'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');


camList = webcamlist
cam = webcam(1)
preview(cam);

%% Acquire a Frame
% To acquire a single frame, use the |snapshot| function.
img = snapshot(cam);

% Display the frame in a figure window.
image(img);

%% Acquiring Multiple Frames
% A common task is to repeatedly acquire a single image, process it, and
% then store the result. To do this, |snapshot| should be called in a loop. 

for idx = 1:200
    img = snapshot(cam);
    image(img);
    img=imresize(img,[227 ])
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),img);

layer = 'fc7';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

classifier = fitcecoc(featuresTrain,YTrain);

YPred = predict(classifier,featuresTest);
label = YPred(idx(i));
imshow(I);
title(char(label));

accuracy = mean(YPred == YTest);
disp(accuracy);

clear cam
clc;
end