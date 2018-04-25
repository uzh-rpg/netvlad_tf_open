% Need to convert all classes into structs so that python can read them.

load('netvlad/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.mat')
for i = 1:numel(net.layers)
    net.layers{i} = struct(net.layers{i}(1));
end
save('structed.mat', 'net');