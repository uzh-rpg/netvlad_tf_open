function [precision, recall, area_under_curve] = evaluate_pr(...
    gt_positions, closest_match_idcs, closest_match_dists2, ...
    closest_gt_dists2, gt_radius)
% Positions row major.

% Given matches, with each match having a score (distance), precision and 
% recall values are evaluated at different distance thresholds for
% accepting a match. We can very simply evaluate this for all relevant
% thresholds by first sorting the matches by distance: We will trace the PR
% curve by first accepting no matches. Then, we "increase the threshold"
% such that all matches are selected one by one.
[~, sort_i] = sort(closest_match_dists2, 'ascend');

% For each match, we now see whether it would be a true or false positive
% if selected.
matched_pos = gt_positions(closest_match_idcs(sort_i), :);
retr_dists2 = sum((gt_positions(sort_i, :) - matched_pos).^2, 2);
tp_on_select = retr_dists2 < gt_radius^2;
fp_on_select = ~tp_on_select;
% By performing the cumulative sum, we obtain true and false positives for
% each of the "threshold increase" steps.
tp_cs = cumsum(tp_on_select);
fp_cs = cumsum(fp_on_select);

% False negatives are a bit trickier. First, the full relevant set are
% false negatives. As we increase the threshold, true positive matches will
% subtract from false negatives.
relevant_size = nnz(closest_gt_dists2(sort_i) < gt_radius^2);
fn_rcs = relevant_size - tp_cs;

precision = tp_cs ./ (tp_cs + fp_cs + 1e-12);
recall = tp_cs ./ (tp_cs + fn_rcs + 1e-12);

% Const interpolation.
area_under_curve = sum(...
    (recall(2:end) - recall(1:(end-1))) .* precision(2:end));

end
