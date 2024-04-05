from torch_geometric.data import Data
from torch_geometric.explain import Explainer, PGExplainer
import torch
from gnn.molgraph_data import MolGraphDataset, molgraph_collate_fn, PyGGraphDataset, pyggraph_collate_fn
from torch.utils.data import DataLoader
from train import MODEL_CONSTRUCTOR_DICTS
import argparse
from captum.attr import Saliency, IntegratedGradients
import numpy as np
import random
from collections import defaultdict

#
# dataset = ...
# loader = DataLoader(dataset, batch_size=1, shuffle=True)


common_args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

common_args_parser.add_argument('--cuda', action='store_true', default=False, help='Enables CUDA training')

common_args_parser.add_argument('--train-set', type=str, default='data/ESOL_train.csv.gz', help='Training dataset path')
common_args_parser.add_argument('--valid-set', type=str, default='data/ESOL_valid.csv.gz', help='Validation dataset path')
common_args_parser.add_argument('--test-set', type=str, default='data/ESOL_test.csv.gz', help='Testing dataset path')
# common_args_parser.add_argument('--loss', type=str, default='MaskedMultiTaskCrossEntropy', choices=[k for k, v in LOSS_FUNCTIONS.items()])
common_args_parser.add_argument('--score', type=str, default='RMSE', help='roc-auc or MSE')

# common_args_parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
# common_args_parser.add_argument('--batch-size', type=int, default=50, help='Number of graphs in a mini-batch')
# common_args_parser.add_argument('--learn-rate', type=float, default=1e-5)

# common_args_parser.add_argument('--savemodel', action='store_true', default=False, help='Saves model with highest validation score')


main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = main_parser.add_subparsers(help=', '.join([k for k, v in MODEL_CONSTRUCTOR_DICTS.items()]), dest='model')
subparsers.required = True

model_parsers = {}
for model_name, constructor_dict in MODEL_CONSTRUCTOR_DICTS.items():
    subparser = subparsers.add_parser(model_name, parents=[common_args_parser])
    for hp_name, hp_kwargs in constructor_dict['hyperparameters'].items():
        subparser.add_argument('--' + hp_name, **hp_kwargs, help=model_name + ' hyperparameter')
    model_parsers[model_name] = subparser


def explain(method, data, args, target=0):
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True)
    if args.cuda:
        input_mask = input_mask.cuda()
    if method == 'ig':
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask, target=target,
                            additional_forward_args=(data,),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        saliency = Saliency(model_forward)
        mask = saliency.attribute(input_mask, target=target,
                                  additional_forward_args=(data,))
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:  # avoid division by zero
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask


def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict


def model_forward(edge_mask, data):
    batch = torch.zeros(data.x.shape[0], dtype=int)
    out = model(data.x, data.edge_index, batch, edge_mask)
    return out

def main():
    # PGExplainer needs to be trained separately since it is a parametric
    # explainer i.e it uses a neural network to generate explanations:
    args = main_parser.parse_args()
    args_dict = vars(args)
    model_hp_kwargs = {
        name.replace('-', '_'): args_dict[name.replace('-', '_')]   # argparse converts to "_" implicitly
        for name, v in MODEL_CONSTRUCTOR_DICTS[args.model]['hyperparameters'].items()
    }

    # train_model_dataset = MolGraphDataset(args.train_set)


    train_dataset = PyGGraphDataset(args.train_set)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=pyggraph_collate_fn)
    validation_dataset = PyGGraphDataset("data/ESOL_valid.csv.gz")
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, collate_fn=pyggraph_collate_fn)
    test_dataset = PyGGraphDataset("data/ESOL_test.csv.gz")
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pyggraph_collate_fn)

    (sample_adjacency, sample_nodes, sample_edges, sample_edge_index, sample_edge_attr), sample_targets = train_dataset[0]


    model = MODEL_CONSTRUCTOR_DICTS[args.model]['constructor'](
        node_features=len(sample_nodes[0]), edge_features=len(sample_edges[0, 0]), out_features=len(sample_targets),
        **model_hp_kwargs
    )
    # load model
    model.load_state_dict(torch.load("savedmodels/GGNN_ESOL_predicted_log_solubility.pth"))
    if args.cuda:
        model = model.cuda()

    data = random.choice([t for t in test_dataset])
    mol = to_molecule(data)

    for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
        edge_mask = explain(method, data, target=0)
        edge_mask_dict = aggregate_edge_directions(edge_mask, data)
        plt.figure(figsize=(10, 5))
        plt.title(title)
        draw_molecule(mol, edge_mask_dict)




    # explainer = Explainer(
    #     model=model,
    #     algorithm=PGExplainer(epochs=30, lr=0.003),
    #     explanation_type='phenomenon',
    #     edge_mask_type='object',
    #     model_config=dict(
    #         mode='regression',
    #         task_level='graph',
    #         return_type='raw',
    #     ),
    #     # Include only the top 10 most important edges:
    #     threshold_config=dict(threshold_type='topk', value=10),
    # )
    #
    #
    # # for epoch in range(30):
    # #     for batch in loader:
    # #         loss = explainer.algorithm.train(
    # #             epoch, model, batch.x, batch.edge_index, target=batch.target)
    #
    # # Generate the explanation for a particular graph:
    #
    #
    # for epoch in range(30):
    #     # explainer.train()
    #     for i_batch, batch in enumerate(train_dataloader):
    #
    #         (adjacency, nodes, edge_index, edge_attr), targets = batch
    #         if args.cuda:
    #             nodes = nodes.cuda()
    #             edge_index = edge_index.cuda()
    #             edge_attr = edge_attr.cuda()
    #             targets = targets.cuda()
    #         loss = explainer.algorithm.train(epoch, model=model, x=nodes, edge_index=edge_index, target=targets, index=None)
    #         print(loss)
    #
    #
    # explanation = explainer(sample_nodes, sample_edge_index, target=sample_targets)
    # print(explanation.edge_mask)

    # Train against a variety of node-level or graph-level predictions:




    # Get the final explanations:

if __name__ == '__main__':
    # config = {
    #     'epochs': 30,
    #     'lr': 0.003,
    #     'batch_size': 1,
    #     'shuffle': True,
    #     'collate_fn': molgraph_collate_fn,
    # }

    main()
