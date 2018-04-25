%% Init
% Requires symlink ../kitti which points to the KITTI dataset root.
netvlad_dim = 4096;
use_dim = 128;
assert(use_dim <= netvlad_dim);

%% Init net
net_file = 'netvlad/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.mat';
load(net_file, 'net');
net= relja_simplenn_tidy(net);

%% Get KITTI feats if not already present.
sequence_id = '00';
kitti_feats_file = ['../kitti/' sequence_id '/netvlad_feats.bin'];
if ~exist(kitti_feats_file, 'file')
    serialAllFeats(net, './', ...
        files_with_ext(['../kitti/' sequence_id '/image_0'], 'jpg'), ...
        kitti_feats_file, 'batchSize', 4);
end
kitti_feats = reshape(fread(...
    fopen(kitti_feats_file, 'rb'), inf, 'float32=>single'), ...
    netvlad_dim, []);
kitti_feats = kitti_feats(1:use_dim, :);

%% Netvlad matching, reject along diagonal (i.e. query ~ match):
sq_dists = squareform(pdist(kitti_feats', 'squaredeuclidean'));
% Suppressing a certain radius around the diagonal to prevent self-matches
% (not real loop closures).
suppression_diameter = 501
diag_suppression = conv2(eye(size(sq_dists, 1)), ...
    ones(1, suppression_diameter), 'same');
sq_dists(diag_suppression > 0) = Inf;
figure(1);
imagesc(sq_dists);
colorbar();
title('Confusion matrix NetVLAD');
[nv_dists2, nv_indices] = min(sq_dists);

%% Ground truth matching:
kitti_poses = load('../kitti/poses/00.txt');
kitti_positions = kitti_poses(:, [4 8 12]);

sq_dists = squareform(pdist(kitti_positions, 'squaredeuclidean'));
sq_dists(diag_suppression > 0) = Inf;
figure(2);
imagesc(sqrt(sq_dists));
colorbar();
title('Confusion matrix GPS');
[gt_dists2, gt_indices] = min(sq_dists);

%% Evaluate
gt_radius = 5;
[precision, recall, auc] = evaluate_pr(...
    kitti_positions, nv_indices, nv_dists2, gt_dists2, gt_radius);
figure(3);
plot(recall, precision);
xlabel('recall');
ylabel('precision');
xlim([0, 1]);
ylim([0, 1]);
title(['PR curve, AUC = ' num2str(auc)]);

save('kitti_pr.mat', 'precision', 'recall', 'auc')
