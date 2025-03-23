from sklearn.preprocessing import StandardScaler


import numpy as np

import data_utils as du
import models
import dill
import pandas as pd


def run(args, grid_eval=False, test_eval_vis=False):
    random_state = args.random_state

    rng = np.random.default_rng(random_state)
    
    print(f"--- Preparing data")

    X_train, X_full, Y_train, Y_full = du.load_dataset(dataset=args.dataset, return_type="X_y",test_split=args.test_split, random_state=random_state)
    df = du.load_dataset(args.dataset, test_split=args.test_split, random_state=random_state)
    print(len(X_train),len(X_full))
    df_test = df.copy()
    assert "S_flip" in df.columns, df.columns
    assert "is_test" in df.columns, df.columns
    
    if args.scale_data:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_full_scaled = scaler.transform(X_full)
    else:
        X_train_scaled = X_train
        X_full_scaled = X_full

    path_template = du.path_name_from_args(args, eval=False)
    path_template_eval = du.path_name_from_args(args, eval=True)
    
    print(f"--- Training model {args.model_name}")
    mlp_path =  models.train_model(args.model_name,
                                                random_state=random_state,
                                                hidden_layer_sizes=args.hidden_layer_sizes,
                                                activation=args.activation,
                                                X_train = X_train_scaled,
                                                Y_train = Y_train,
                                                smoke_test = args.smoke_test,
                                                pkl_path=f"model_pkl/{path_template}")
    
    print(f"--- Computing predicted probabilities on test split of {args.dataset}")
    # load model
    with open(mlp_path, "rb") as f:
        mlp = dill.load(f)
    if args.ensemble_n ==1 :
        model, predict_func = models.convert_sklearn_mlp_to_hummingbird(mlp, mode="CPU")
    elif args.ensemble_n > 1:
        model, predict_func = models.convert_sklearn_ensemble_to_hummingbird(mlp, mode="CPU")

    # save preds on full dataset
    scores = dict()
    scores["pred_prob"] = predict_func(model,X_full_scaled)[:,1]

    for scoring_key in scores.keys():
        df_test[scoring_key] = scores[scoring_key]
    df_test.to_csv(f"preds_csv/df_test-{path_template}.csv")
    
    if args.dataset != args.dataset_eval:
        print(f"--- Computing predicted probabilities on final eval dataset {args.dataset_eval}")
        # save preds on final eval dataset
        df_vis = du.load_dataset(args.dataset_eval, test_split="none", random_state=random_state)
        X_vis, _ = du.load_dataset(dataset=args.dataset_eval, return_type="X_y", test_split="none")
        
        if args.scale_data:
            X_vis_scaled = scaler.transform(X_vis)
        else:
            X_vis_scaled = X_vis
            
        scores = dict()
        scores["pred_prob"] = predict_func(model,X_vis_scaled)[:,1]

        for scoring_key in scores.keys():
            df_vis[scoring_key] = scores[scoring_key]
        df_vis.to_csv(f"preds_csv/df_test-{path_template_eval}.csv")

    print(f"--- Calculating thresholds on val set of {args.dataset} and running evaluation on final eval dataset {args.dataset_eval}")
    ## find thresholds on val set and eval on final eval dataset
    seed_range = range(10) if args.dataset == "OP_mul" else [0]
    val_ratio = 0.2 if args.dataset == "OP_mul" else 1

    list_of_res_dict = list()
    for seed_val in seed_range:
        res_dict = models.eval_model(args,seed_val=seed_val, threshold_mode=args.threshold_mode, val_ratio=val_ratio,
                              csv_preds=f"preds_csv/df_test-{path_template}.csv")
        if res_dict is not None: list_of_res_dict.append(res_dict)

    df_thresh = pd.DataFrame(list_of_res_dict)
    df_thresh.to_csv(f"results_csv/df_valthresh_{path_template}.csv")
    
    ## visualise predictions on final eval dataset
    if args.dataset == "OP_sin" and test_eval_vis:
        print(f"--- Visualising predictions on final eval dataset {args.dataset_eval}")
        #df_test["is_test"] = True
        assert "S_flip" in df_vis.columns, df_vis.columns
        du.test_eval_visualisation(args, df_vis, random_state=random_state, df_thresh=df_thresh)

if __name__ == "__main__":

    import itertools
    from tqdm import tqdm

    du.check_results_folders()

    SEED = 0
    args = du.Args(random_state=SEED)
    SPLIT = "random_0.8"
    DATASET = "OP_mul"
    SCALING = True
    HIDDEN_SIZE = (64,64)
    ACTIVATION = "sigmoid"

    # ENSEMBLE_N = 1
    # MODEL = "sklearn_mlp"
    
    ENSEMBLE_N = 10
    MODEL = f"sklearn_mlp_ensemble_{ENSEMBLE_N}_0"
    
    THRESHOLD_MODE = "20rejection_or_highprecision"
    
    args.update(
        dataset=DATASET,
        hidden_layer_sizes=HIDDEN_SIZE,
        activation=ACTIVATION,
        scale_data=SCALING,
        test_split=SPLIT,
        model_name=MODEL,
        ensemble_n=ENSEMBLE_N,
        threshold_mode=THRESHOLD_MODE
    )
            
    
    if DATASET == "OP_sin":
        args.update(dataset_eval="OP_sin_more") # final eval on the denser dataset
    elif DATASET == "OP_mul":
        args.update(dataset_eval="OP_mul")
    
    args.print()

    run(args, grid_eval=False, test_eval_vis=True)
