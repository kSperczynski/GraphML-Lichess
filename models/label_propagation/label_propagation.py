from torch_geometric.nn import LabelPropagation
import torch

def label_propagation(y, edge_index, mask, num_layers, alpha):
    """
    Performs label propagation regression on the given graph and labelled nodes.

    y           -> regression targets
    edge_index  -> graph structure
    mask        -> 1 if node is labelled, 0 otherwise
    num_layers  -> number of iterations for label propagation
    alpha       -> reset coefficient

    return predicted targets for each node
    """
    # with reset, LP would be biased towards the rating of 0
    # if the targets are mean-shifted first, the bias is towards the mean rating instead
    mean_target = y[mask].mean()
    normalised_targets = (y - mean_target).view(-1, 1)

    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.LabelPropagation.html#torch_geometric.nn.models.LabelPropagation
    model = LabelPropagation(num_layers=num_layers, alpha=alpha)
    out = model(normalised_targets, edge_index, mask=mask, post_step=lambda x: x)

    return out.squeeze(-1) + mean_target


def hyperparameter_search(y, edge_index, train_mask, val_mask, nums_layers, alphas):
    """
    Performs hyperparameter search (gridsearch) over `alpha` and `num_layers`.

    y           -> regression targets
    edge_index  -> graph structure
    train_mask  -> mask representing the labelled nodes
    val_mask    -> mask representing the validaiton nodes
    nums_layers -> options to try for the `num_layers` hyperparameter
    alphas      -> options to try for the `alpha` hyperparameter

    return best `alpha` and `num_layers` found via grid search, and the best found MSE on the validation set
    """
    best_alpha, best_num_layers, best_val_mse = None, None, float("inf")

    for a in alphas:
        for num_layers in nums_layers:
            pred = label_propagation(y, edge_index, mask=train_mask, num_layers=num_layers, alpha=a)
            val_mse = metrics(pred[val_mask], y[val_mask])["mse"]

            if val_mse < best_val_mse:
                best_alpha, best_num_layers, best_val_mse = a, num_layers, val_mse

    return {
        "best_alpha": best_alpha,
        "best_num_layers": best_num_layers,
        "best_val_mse": best_val_mse
    }


def make_masks(num_nodes, val_frac, test_frac, seed):
    """
    Randomly geberates masks for splitting nodes into test, train, and validation sets.

    num_nodes -> total number of nodes in the graph
    val_frac  -> fraction of nodes to be used for validation
    test_frac -> fraction of nodes to be used for the test set
    seed      -> random seed to guide mask creation

    return test, validation, and train masks
    """
    assert val_frac + test_frac < 1.0

    perm = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed))
    n_val, n_test = int(val_frac * num_nodes), int(test_frac * num_nodes)

    def to_mask(idx):
        m = torch.zeros(num_nodes, dtype=torch.bool)
        m[idx] = True
        return m

    val_mask = to_mask(perm[:n_val])
    test_mask = to_mask(perm[n_val:n_val + n_test])
    train_mask = to_mask(perm[n_val + n_test:])

    return train_mask, val_mask, test_mask


def metrics(pred, target):
    """
    Copmute MSE, MAE, and R2 sof the predicted values given the true targets.
    """
    err = pred - target
    mse = (err ** 2).mean().item()
    mae = err.abs().mean().item()
    r2 = 1.0 - (err ** 2).sum().item() / ((target - target.mean()) ** 2).sum().item()
    return {"mse": mse, "mae": mae, "r2": r2}


def print_LP_evaluation_results(results, val_frac=0.1, train_fracs=[x / 10 for x in range(1, 9)]):
    print("------------------------------------------------------------------")
    print("LABEL PROPAGATION RESULTS:")
    for i, train_frac in enumerate(train_fracs):
        print("------------------------------------------------------------------")
        print(f"TRAIN-VAL-TEST = {train_frac}-{val_frac}-{(1-val_frac-train_frac):.1f}")
        print(f"MSE = {results[i]['mse']}")
        print(f"MAE = {results[i]['mae']}")
        print(f"R2  = {results[i]['r2']}")


def evaluate_label_propagation(y, edge_index, val_frac=0.1, train_fracs=[x / 10 for x in range(1, 9)], nums_layers=[1, 5, 10, 20, 30, 50, 75, 100, 200], alphas=[x / 10 for x in range(1, 11)], seeds=range(5), verbose=False):
    """
    Evaluates the performance of Label Propagation for a range of train set sizes. Performs hyperparameter tuning on `alpha` and `num_layers`.

    y           -> regression targets
    edge_index  -> graph structure
    val_frac    -> fraction of nodes in the validation set
    train_fracs -> range of train set fraction sizes
    nums_layers -> options to try for the `num_layers` hyperparameter
    alphas      -> options to try for the `alpha` hyperparameter
    seeds       -> random seeds used for mask creations; metrics are averages across seeds
    verbose     -> whether to print results of hyperparameter tuning to the terminal

    return MSE, MAE, R2 for all values of of train set sizes
    """
    num_nodes = y.numel()

    res = []

    if verbose:
        print("-------------------------------------------------------")
        print("HYPERPARAMETER TUNING:")

    for train_frac in train_fracs:
        total_mse = 0
        total_mae = 0
        total_r2 = 0

        total_alphas = 0
        total_layers = 0
        total_val_mse = 0

        for s in seeds:
            train_mask, val_mask, test_mask = make_masks(num_nodes, val_frac, 1 - val_frac - train_frac, seed=s)

            hyperparams = hyperparameter_search(y, edge_index, train_mask, val_mask, nums_layers, alphas)
            best_alpha = hyperparams['best_alpha']
            best_num_layers = hyperparams['best_num_layers']
            best_val_mse = hyperparams['best_val_mse']

            total_alphas += best_alpha
            total_layers += best_num_layers
            total_val_mse += best_val_mse

            preds = label_propagation(y, edge_index, train_mask | val_mask, best_num_layers, best_alpha)

            run_metrics = metrics(preds[test_mask], y[test_mask])

            total_mse += run_metrics['mse']
            total_mae += run_metrics['mae']
            total_r2 += run_metrics['r2']

        avg_mse = total_mse / len(seeds)
        avg_mae = total_mae / len(seeds)
        avg_r2 = total_r2 / len(seeds)


        avg_alpha = total_alphas / len(seeds)
        avg_layers = total_layers / len(seeds)
        avg_val_mse = total_val_mse / len(seeds)

        if verbose:
            print("---------------------------------------------------")
            print(f"TRAIN-VAL-TEST = {train_frac}-{val_frac}-{(1-val_frac-train_frac):.1f}")
            print(f"ALPHA = {avg_alpha}")
            print(f"NUM_LAYERS = {avg_layers}")
            print(f"VAL_MSE = {avg_val_mse}")

        res.append({
            "mse": avg_mse,
            "mae": avg_mae,
            "r2": avg_r2
        })

    return res


def trivial_baseline(y):
    mean_target = y.mean()

    res = metrics(mean_target, y)

    print("---------------------------------------------")
    print("Trivial baseline:")
    print(f"MSE = {res['mse']}")
    print(f"MAE = {res['mae']}")
    print(f"R2  = {res['r2']}")