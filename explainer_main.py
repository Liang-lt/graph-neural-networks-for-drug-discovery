from torch_geometric.data import Data
from torch_geometric.explain import Explainer, PGExplainer
from gnn.molgraph_data import MolGraphDataset, molgraph_collate_fn, PyGGraphDataset
from torch.utils.data import DataLoader
from train import MODEL_CONSTRUCTOR_DICTS
import argparse
#
# dataset = ...
# loader = DataLoader(dataset, batch_size=1, shuffle=True)


common_args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

common_args_parser.add_argument('--cuda', action='store_true', default=True, help='Enables CUDA training')

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




def main():
    # PGExplainer needs to be trained separately since it is a parametric
    # explainer i.e it uses a neural network to generate explanations:
    args = main_parser.parse_args()
    args_dict = vars(args)
    model_hp_kwargs = {
        name.replace('-', '_'): args_dict[name.replace('-', '_')]   # argparse converts to "_" implicitly
        for name, v in MODEL_CONSTRUCTOR_DICTS[args.model]['hyperparameters'].items()
    }

    train_model_dataset = MolGraphDataset(args.train_set)


    train_dataset = PyGGraphDataset(args.train_set)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_dataset = PyGGraphDataset("data/ESOL_valid.csv.gz")
    validation_dataloader = DataLoader(validation_dataset, batch_size=1)
    test_dataset = PyGGraphDataset("data/ESOL_test.csv.gz")
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    (sample_adjacency, sample_nodes, sample_edge_index, sample_edge_attr), sample_targets = train_dataset[0]
    (adjacency0, nodes0, edges0), targets0 = train_model_dataset[0]


    model = MODEL_CONSTRUCTOR_DICTS[args.model]['constructor'](
        node_features=len(sample_nodes[0]), edge_features=len(edges0[0, 0]), out_features=len(sample_targets),
        **model_hp_kwargs
    )
    if args.cuda:
        model = model.cuda()

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=30, lr=0.003),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
        # Include only the top 10 most important edges:
        threshold_config=dict(threshold_type='topk', value=10),
    )


    # for epoch in range(30):
    #     for batch in loader:
    #         loss = explainer.algorithm.train(
    #             epoch, model, batch.x, batch.edge_index, target=batch.target)

    # Generate the explanation for a particular graph:


    for epoch in range(30):
        # explainer.train()
        for i_batch, batch in enumerate(train_dataloader):

            (adjacency, nodes, edge_index, edge_attr), targets = batch
            if args.cuda:
                nodes = nodes.cuda()
                edge_index = edge_index.cuda()
                edge_attr = edge_attr.cuda()
                targets = targets.cuda()
            loss = explainer.algorithm.train(epoch, model=model, x=nodes, edge_index=edge_index, target=targets, index=None)
            print(loss)


    explanation = explainer(sample_nodes, sample_edge_index, target=sample_targets)
    print(explanation.edge_mask)

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
