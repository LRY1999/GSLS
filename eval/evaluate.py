import scipy
import numpy as np
from scipy.sparse import diags
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from munkres import Munkres
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

# "from https://github.com/Ruiqi-Hu/ARGA"
class clustering_metrics():
    def __init__(self, true_label, predict_label):

        self.true_label = true_label
        self.pred_label = predict_label
        self.true_label_onehot = true_label.reshape(-1,1)
        self.one_hot=OneHotEncoder(sparse=False)
        self.true_label_onehot= self.one_hot.fit_transform(self.true_label_onehot)


    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)


        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)
        new_predict = np.zeros(len(self.pred_label))

        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]

            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        pred_label_onehot = new_predict.reshape(-1, 1)
        pred_label_onehot = self.one_hot.fit_transform(pred_label_onehot)
        # metrics.accuracy_score
        # metrics.balanced_accuracy_score
        # metrics.top_k_accuracy_score
        # metrics.average_precision_score
        # metrics.brier_score_loss Brier
        # metrics.f1_score   F1score
        # metrics.log_loss
        # metrics.precision_score
        # metrics.recall_score
        # metrics.jaccard_score
        # metrics.roc_auc_score
        # metrics.cohen_kappa_score

        acc = metrics.accuracy_score(self.true_label, new_predict)

        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        log_loss=metrics.log_loss(self.true_label_onehot, pred_label_onehot, eps=1e-15)
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro,log_loss

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, log_loss= self.clusteringAcc()

        return acc, nmi, adjscore


#"from https://github.com/tkipf/gae"
def get_roc_score(edges_pos, edges_neg, embeddings, adj_sparse):
    score_matrix = np.dot(embeddings, embeddings.T)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))  # predicted score
        pos.append(adj_sparse[edge[0], edge[1]])  # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))  # predicted score
        neg.append(adj_sparse[edge[0], edge[1]])  # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    # print(preds_all, labels_all )

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score




def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not scipy.sparse.isspmatrix_coo(mx): #是否为csr_matrix类型
            mx = mx.tocoo() #实现csc矩阵转换为coo矩阵
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            ## np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组，堆叠的数组需要具有相同的维度，transpose()作用是转置
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

