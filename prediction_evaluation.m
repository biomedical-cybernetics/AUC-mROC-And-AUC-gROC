function measures = prediction_evaluation(scores, labels)

%%% INPUT %%%
% scores - numerical scores for the samples
% labels - binary labels indicating the positive and negative samples

%%% OUTPUT %%%
% measures - structure containing the following fields:
%   prec      - precision
%   auc_prec  - area under precision curve
%   auc_pr    - area under precision-recall curve
%   auc_roc   - area under roc curve
%   auc_mroc  - area under m-roc curve
%   ndcg - normalized discounted cumulative gain
%   mcc - matthews correlation coefficient

validateattributes(scores, {'numeric'}, {'vector','finite'})
S = length(scores);
validateattributes(labels, {'numeric'}, {'vector','binary','numel',S})
P = full(sum(labels==1));
N = S - P;
if P==0 || N==0
    error('labels cannot be all ones or all zeros')
end
if isrow(scores); scores = scores'; end
if isrow(labels); labels = labels'; end

measures = compute_curves_measures(scores, labels, S, P, N);
measures.ndcg = compute_ndcg(scores, labels, P);
measures.mcc = compute_mcc(scores, labels, P, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function measures = compute_curves_measures(scores, labels, S, P, N)

[scores,idx] = sort(-scores, 'ascend');
labels = labels(idx);
[~,ut,~] = unique(scores);
ut = [ut(2:end)-1; S];

tp = full(cumsum(labels));
fp = full(cumsum(~labels));
tp_rand = fp .* (P/N);

prec = tp./(1:S)';
tpr = tp/P;
fpr = fp/N;
tpr_m = log(1+tp)/log(1+P);
fpr_m = log(1+fp)/log(1+N);
tpr_m_rand = log(1+tp_rand)/log(1+P);
tpr_m_norm = (tpr_m-tpr_m_rand) ./ (1-tpr_m_rand) .* (1-fpr_m) + fpr_m; tpr_m_norm(isnan(tpr_m_norm)) = 1;

measures.prec = prec(P);
if P==1
    measures.auc_prec = prec(1);
else
    measures.auc_prec = trapz(1:P,prec(1:P)) / (P-1);
end

prec = prec(ut);
tpr = [0; tpr(ut)];
fpr = [0; fpr(ut)];
fpr_m = [0; fpr_m(ut)];
tpr_m_norm = [0; tpr_m_norm(ut)];

if all(tpr(2:end)==1)
    measures.auc_pr = prec(1);
else
    measures.auc_pr = trapz(tpr(2:end),prec) / (1-tpr(2));
end
measures.auc_roc = trapz(fpr,tpr);
measures.auc_mroc = trapz(fpr_m,tpr_m_norm);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ndcg = compute_ndcg(scores, labels, P)

ranks = tiedrank(-scores);
dcg = sum(1./log(1+ranks(labels==1)));
idcg = sum(1./log(1+(1:P)));
ndcg = dcg / idcg;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function mcc = compute_mcc(scores, labels, P, N)

[~,idx] = sort(scores, 'descend');
labels = labels(idx);

tp = full(sum(labels(1:P)));
fp = P - tp;
tn = N - fp;
fn = fp;

mcc = (tp*tn - fp*fn) / sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn));
if isinf(mcc) || isnan(mcc)
    mcc = 0;
end
