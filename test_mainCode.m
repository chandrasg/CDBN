%% Layer 1
images = [];
dataname='olshausen';
images_all = sample_images_all(dataname);
% images_all=images;

ws=10; %Weight shape Nw
num_bases= 24; %Number of weights K
pbias=.003; % sparsity parameter - expected activations of the hidden units
pbias_lb=.01; %Not used
pbias_lambda=5; %Learning rate for sparsity update
spacing=2; %C - pooling ratio
epsilon=.02; % Learning rate - the ratio of change taken from gradient descent
l2reg=.01; % Weight decay - regularization factor to keep the length of weight vector small
batch_size=10; % Number of patches from each image

train_tirbm_updown_LB_v1h(images_all, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size);
% clearvars '*' -except W pars t vbias_vec hbias_vec error_history sparsity_history ws num_bases pbias pbias_lb pbias_lambda spacing epsilon l2reg batch_size 
load('layer1.mat');

W1=W;
pars1=pars;
vbias_vec1=vbias_vec;
hbias_vec1=hbias_vec;
error_history1=error_history;
sparsity_history1=sparsity_history;

figure(1), display_network(W1);
saveas(gcf, sprintf('Layer1_Visualization.png'));
dataname='olshausen';
ConvolvedFeaturesL1=[];
images_all = sample_images_conv(dataname);

for i=1:numel(images_all)

imdata_batch=images_all{i};    
imdata_batch = imdata_batch - mean(imdata_batch(:));
imdata_batch = trim_image_for_spacing_fixconv(imdata_batch, ws, spacing);
poshidexp = tirbm_inference(imdata_batch, W1, hbias_vec1, pars1);
[poshidstates poshidprobs] = tirbm_sample_multrand2(poshidexp, spacing);
ConvolvedFeaturesL1=cat(4,ConvolvedFeaturesL1,poshidstates);
end

% [PooledFeaturesL1,idxl1] = MaxPooling(ConvolvedFeaturesL1, [2 2]);%Replace by Pool!
ConvolvedFeaturesL1=permute(ConvolvedFeaturesL1,[3,4,1,2]);
PooledFeaturesL1=Pool(2,ConvolvedFeaturesL1);
PooledFeaturesL1=permute(PooledFeaturesL1,[3,4,1,2]);

%% Layer 2

numImages=size(PooledFeaturesL1,4);
images_all_conv=cell(1,numImages);
for i=1:numImages

    images_all_conv{i}=squeeze(PooledFeaturesL1(:,:,:,i));
    
end

ws=10;
num_bases= 40;
pbias=.005;
pbias_lb=5;
pbias_lambda=5;
spacing=2;
epsilon=.005;
l2reg=.01;
batch_size=2;

layer2_train_tirbm_updown_LB_v1h(images_all_conv, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size,W1);
% clearvars '*' -except W pars t vbias_vec hbias_vec error_history sparsity_history W1 pars1 t1 vbias_vec1 hbias_vec1 error_history1 sparsity_history1 ws num_bases pbias pbias_lb pbias_lambda spacing epsilon l2reg batch_size
load('layer2.mat');

W2=W;
pars2=pars;
vbias_vec2=vbias_vec;
hbias_vec2=hbias_vec;
error_history2=error_history;
sparsity_history2=sparsity_history;

figure(1), vis_layer2=display_network_layer2(W2,W1);
saveas(gcf, sprintf('Layer2_Visualization.png'));

% load layer2.mat
ConvolvedFeaturesL2=[];
for i=1:numel(images_all_conv)

imdata_batch=images_all_conv{i};    
imdata_batch = imdata_batch - mean(imdata_batch(:));
imdata_batch = trim_image_for_spacing_fixconv(imdata_batch, ws, spacing);
poshidexp = tirbm_inference(imdata_batch, W2, hbias_vec2, pars2);
[poshidstates poshidprobs] = tirbm_sample_multrand2(poshidexp, spacing);
ConvolvedFeaturesL2=cat(4,ConvolvedFeaturesL2,poshidstates);
end
ConvolvedFeaturesL2=permute(ConvolvedFeaturesL2,[3,4,1,2]);
PooledFeaturesL2=Pool(2,ConvolvedFeaturesL2);
PooledFeaturesL2=permute(PooledFeaturesL2,[3,4,1,2]);
%%Layer3

numImages=size(PooledFeaturesL2,4);
images_all_conv=cell(1,numImages);
for i=1:numImages

    images_all_conv{i}=squeeze(PooledFeaturesL2(:,:,:,i));
    
end

ws=14;
num_bases= 24;
pbias=.005;
pbias_lb=10;
pbias_lambda=10;
spacing=2;
epsilon=0.00005;
l2reg=.05;
batch_size=2;

layer3_train_tirbm_updown_LB_v1h(images_all_conv, ws, num_bases, pbias, pbias_lb, pbias_lambda, spacing, epsilon, l2reg, batch_size,vis_layer2);
% clearvars '*' -except W pars t vbias_vec hbias_vec error_history sparsity_history W1 pars1 t1 vbias_vec1 hbias_vec1 error_history1 sparsity_history1 W2 pars2 t2 vbias_vec2 hbias_vec2 error_history2 sparsity_history2 ws num_bases pbias pbias_lb pbias_lambda spacing epsilon l2reg batch_size
load('layer3.mat');

W3=W;
pars3=pars;
vbias_vec3=vbias_vec;
hbias_vec3=hbias_vec;
error_history3=error_history;
sparsity_history3=sparsity_history;

figure(1), display_network_layer2(W3,vis_layer2);
saveas(gcf, sprintf('Layer3_Visualization.png'));
