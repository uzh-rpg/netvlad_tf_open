%% Load net (separate section because expensive)
load('netvlad/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.mat')

%% Generate layer outputs and measure time
net = relja_simplenn_tidy(net);
net = relja_simplenn_move(net, 'cpu');

ims_ = vl_imreadjpeg({'../example.jpg'});
ims = cat(4, ims_{:});
ims(:,:,1,:)= ims(:,:,1,:) - net.meta.normalization.averageImage(1,1,1);
ims(:,:,2,:)= ims(:,:,2,:) - net.meta.normalization.averageImage(1,1,2);
ims(:,:,3,:)= ims(:,:,3,:) - net.meta.normalization.averageImage(1,1,3);

simpleNnOpts= {'conserveMemory', false, 'mode', 'test'};

vl_simplenn(net, ims, [], [], simpleNnOpts{:});
vl_simplenn(net, ims, [], [], simpleNnOpts{:});
tic;
stats = vl_simplenn(net, ims, [], [], simpleNnOpts{:});
toc

outs = {stats.x};
times = {stats.time};

%% Export status to mat file
save('example_stats.mat', 'outs', 'times');