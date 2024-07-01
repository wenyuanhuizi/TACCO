import json
import os
import argparse
import torch.nn as nn
from tqdm import trange
from models import SetGNN
from preprocessing import *
from datetime import datetime
from convert_datasets_to_pygDataset import dataset_Hypergraph
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, epoch, method, dname, args):
    model.eval()

    out_score_g_logits, edge_feat, node_feat, weight_tuple = model(data)
    out_g = torch.sigmoid(out_score_g_logits)

    valid_acc, valid_auroc, valid_aupr, valid_f1_macro = eval_func(data.y[split_idx['valid']],
                                                                   out_g[split_idx['valid']], epoch, method, dname,
                                                                   args, threshold=args.threshold)
    test_acc, test_auroc, test_aupr, test_f1_macro = eval_func(data.y[split_idx['test']],
                                                               out_g[split_idx['test']],
                                                               epoch, method, dname, args,
                                                               threshold=args.threshold)

    return valid_acc, valid_auroc, valid_aupr, valid_f1_macro, \
           test_acc, test_auroc, test_aupr, test_f1_macro

def eval_ukb(y_true, y_pred, epoch, method, dname, args, threshold=0.5):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    pred = np.array(y_pred > threshold).astype(int)
    correct = (pred == y_true)

    accuracy = correct.sum() / correct.size
    f1_macro = f1_score(y_true, pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)

    return accuracy, roc_auc, aupr, f1_macro

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dname', default='ukb')
    parser.add_argument('--method', default='AllSetTransformer')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--cuda', default='1', type=str)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-3, type=float)
    parser.add_argument('--LearnFeat', action='store_true')
    parser.add_argument('--All_num_layers', default=1, type=int)  # hyperparameter L
    parser.add_argument('--MLP_num_layers', default=1, type=int)
    parser.add_argument('--MLP_hidden', default=48, type=int)  # hyperparameter d
    parser.add_argument('--Classifier_num_layers', default=2, type=int)
    parser.add_argument('--Classifier_hidden', default=64, type=int)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--normtype', default='all_one')  # ['all_one','deg_half_sym']
    parser.add_argument('--add_self_loop', action='store_false')
    parser.add_argument('--normalization', default='ln')  # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--num_features', default=0, type=int)  # placeholder
    parser.add_argument('--num_labels', default=25, type=int)  # Adjust based on your dataset
    parser.add_argument('--num_nodes', default=100, type=int)  # Adjust based on your dataset
    parser.add_argument('--feature_dim', default=128, type=int)  # node embedding dim (*2 if use text)
    parser.add_argument('--PMA', action='store_true')
    parser.add_argument('--heads', default=1, type=int)  # attention heads
    parser.add_argument('--output_', default=1, type=int)  # Placeholder
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--remain_percentage', default=0.3, type=float)
    parser.add_argument('--test', default=0, type=int)  # not to use text info in HypEHR

    parser.set_defaults(PMA=True)
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(LearnFeat=True)

    args = parser.parse_args()

    dname = args.dname
    p2raw = '../data/raw_data/'
    dataset = dataset_Hypergraph(name=dname, root='../data/pyg_data/hypergraph_dataset/',
                                 p2raw=p2raw, num_nodes=args.num_nodes)
    data = dataset.data
    args.num_features = dataset.num_features

    # Shift the y label to start with 0
    data.y = data.y - data.y.min()

    if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
    if not hasattr(data, 'num_hyperedges'):
        data.num_hyperedges = torch.tensor(
            [data.edge_index[0].max() - data.n_x[0] + 1])

    if args.method == 'AllSetTransformer':
        data = ExtractV2E(data)
        data = norm_contruction(data, option=args.normtype)

    # Custom train-valid-test split
    split_idx = {
        'train': torch.arange(0, 1304),  # 1140 basic + 164 extra (basic features part)
        'valid': torch.arange(1304, 1304 + 81),  # next 81 rows (basic features part)
        'test': torch.arange(1304 + 81, 1304 + 81 + 244)  # next 244 rows (basic features part)
    }

    # hypergraph transformer
    model = SetGNN(args, data)

    # put things to device
    if args.cuda != '-1':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model, data = (model.to(device), data.to(device))

    criterion = nn.BCELoss()

    model.train()
    model.reset_parameters()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training logs
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('../logs/', current_time)
    os.makedirs(log_dir, exist_ok=True)

    # start training
    with torch.autograd.set_detect_anomaly(True):
        for epoch in trange(args.epochs):

            model.train()
            model.zero_grad()

            out_score_logits, out_edge_feat, out_node_feat, _ = model(data)
            out = torch.sigmoid(out_score_logits)

            # classifier loss
            cls_loss = criterion(out[split_idx['train']], data.y[split_idx['train']])

            # final loss
            model_loss = cls_loss

            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            model_optimizer.step()

            eval_function = eval_ukb

            valid_acc, valid_auroc, valid_aupr, valid_f1_macro, \
                test_acc, test_auroc, test_aupr, test_f1_macro = \
                evaluate(model, data, split_idx, eval_function, epoch, args.method, args.dname, args)

            # training logs
            fname_valid = f'{args.dname}_valid_{args.method}.txt'
            fname_test = f'{args.dname}_test_{args.method}.txt'

            fname_valid = os.path.join(log_dir, fname_valid)
            fname_test = os.path.join(log_dir, fname_test)
            fname_hyperparameters = os.path.join(log_dir, 'hyperparameters.txt')

            # save hyperparams
            with open(fname_hyperparameters, 'w', encoding='utf-8') as f:
                args_dict = vars(args)
                f.write(json.dumps(args_dict, indent=4))

            # valid set
            with open(fname_valid, 'a+', encoding='utf-8') as f:
                f.write('Epoch: {}, ACC: {:.5f}, AUROC: {:.5f}, AUPR: {:.5f}, F1_MACRO: {:.5f}\n'
                    .format(epoch + 1, valid_acc, valid_auroc, valid_aupr, valid_f1_macro))

            # test set
            with open(fname_test, 'a+', encoding='utf-8') as f:
                f.write('Epoch: {}, ACC: {:.5f}, AUROC: {:.5f}, AUPR: {:.5f}, F1_MACRO: {:.5f}\n'
                    .format(epoch + 1, test_acc, test_auroc, test_aupr, test_f1_macro))

    print(f'Training finished. Logs are saved in {log_dir}.')
