import scipy.io
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from plotnine import *
import os

class Args():
    def __init__(self, model_name=None, random_state=None,
                 hidden_layer_sizes=(50,50),
                 activation="tanh",
                 dataset="OP_sin",
                 imbalance_strategy="none",
                 smoke_test=False,
                 test_split="random_0.5",
                 scale_data=True,
                 dataset_eval=None):
        self.model_name = model_name
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dataset = dataset
        self.dataset_eval = dataset_eval if dataset_eval is not None else dataset
        self.imbalance_strategy = imbalance_strategy
        self.smoke_test = smoke_test
        self.test_split = test_split
        self.scale_data = scale_data
        self.random_state = random_state
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def print(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

    def to_dict(self):
        return self.__dict__

def check_results_folders():
    for folder_name in ["model_pkl", "preds_csv", "results_csv", "plots","slices", "FP_csv"]:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

def get_path_template(dataset="OP_sin", imbalance_strategy="none", model="sklearn_mlp_mle",
                      hidden_layer_sizes=(64,64), activation="sigmoid", train_size=0.5, normalize_input=False):
    normalize_input_str = "normalized" if normalize_input else "not_normalized"
    return f"{dataset}-{imbalance_strategy}-{model}-{hidden_layer_sizes}-{activation}-trainsize{train_size}-{normalize_input_str}"

def get_dataset_feature_list(dataset="OP_sin", select_features="all", specific_points=None):
    if dataset == "OP_sin" or dataset == "OP_sin_more":
        features = ['V','P','Q']
    elif dataset == "OP_mul":
        features = ['V','P1','Q1','P2','Q2']
    elif dataset == "specific_points":
        features = list(specific_points[0].keys())

    if select_features == "2D":
        features = ['P','Q'] if dataset == "OP_sin" else ['P1','Q1']

    return features

def subset_df(df, selection_dict, mode="include"):
    for column, possible_values in selection_dict.items():
        if not isinstance(possible_values, list):
            possible_values = [possible_values]
        if not column in df.columns:
            print(f"Column {column} not in dataframe")
            continue
        if mode == "include":
            df = df[df[column].isin(possible_values)]
        elif mode == "exclude":
            df = df[~df[column].isin(possible_values)]
        elif mode == "range":
            df = df[(df[column] >= possible_values[0]) & (df[column] <= possible_values[1])]
    return df

def raw_data_to_df(dataset, specific_points=None):
    """Given a dataset name, return a pandas dataframe with the features and label (S) as columns

    Args:
        dataset (str): Dataset name (OP_sin or OP_mul)

    Returns:
        pandas.DataFrame: dataframe with the features and label (S) as columns
    """
    
    #print(f"Loaded mat array with shape",mat[dataset].shape)
    if dataset == "OP_sin":
        data_path = f"Data_VSC/{dataset}.mat"
        mat = scipy.io.loadmat(data_path)    
        df = pd.DataFrame({
            'V': mat[dataset][0],
            'P': mat[dataset][1],
            'Q': mat[dataset][2],
            'S': mat[dataset][3].astype(int)
        })
    elif dataset == "OP_mul":
        data_path = f"Data_VSC/{dataset}.mat"
        mat = scipy.io.loadmat(data_path)
        df = pd.DataFrame({
            'V': mat[dataset][0],
            'P1': mat[dataset][1],
            'Q1': mat[dataset][2],
            'P2': mat[dataset][3],
            'Q2': mat[dataset][4],
            'S': mat[dataset][5].astype(int)
        })
    elif dataset == "OP_sin_more":
        data_path = f"Data_VSC/{dataset}.csv"
        # read csv file without pandas
        data = dict()
        with open(data_path, 'r') as f:
            lines = f.readlines()
            data["V"] = np.array(lines[0].strip().split(",")).astype(float)
            data["P"] = np.array(lines[1].strip().split(",")).astype(float)
            data["Q"] = np.array(lines[2].strip().split(",")).astype(float)
            data["S"] = np.array(lines[3].strip().split(",")).astype(int)
        df_full = pd.DataFrame(data)
        df_orig = sparsify_single_dataset(df_full, step_V=5, step_P=10, step_Q=10)
        df = pd.merge(df_full, df_orig, on=["V", "P", "Q", "S"], how="outer", indicator=True)
        df = df.loc[df["_merge"] == "left_only"].drop("_merge", axis=1)
        #print(f"length of df: {len(df)}")
        df = df.loc[~df["V"].isin(df_orig["V"].unique())]
        #print(f"length of df after removing orig V values: {len(df)}")

    elif dataset == "specific_points":
        df = pd.DataFrame(specific_points)
        print(df)
        return df
    #print(f"Converted to dataframe")
    #print(df.head())
    df['S_flip'] = 1 - df['S']
    return df

def dataset_df_to_Xy(df, features):
    X = np.stack([df[f] for f in features]).swapaxes(1,0)
    #print(f"Input data of shape", X.shape)
    try:
        y = df['S_flip'].to_numpy()
        return X,y
    except:
        return X, None

def path_name_from_args(args, exclude_model=False, eval=False):
    scaling_str = "scaled" if args.scale_data else "unscaled"
    smoke_str = "smoketest_" if args.smoke_test else ""
    dataset = args.dataset if not eval else args.dataset_eval
    if exclude_model:
        path_template = f"{smoke_str}{dataset}-{scaling_str}-{args.hidden_layer_sizes}-{args.activation}-{args.test_split}_split"
    else:
        path_template = f"{smoke_str}{dataset}-{scaling_str}-{args.model_name}-{args.hidden_layer_sizes}-{args.activation}-{args.test_split}_split-seed{args.random_state}"
    return path_template

def sparsify_single_dataset(df, step_V=5, step_P=10, step_Q=10, start_V=0):
    unique_V = df["V"].unique()
    unique_V.sort()
    n_points_per_V = dict()

    selected_V = unique_V[start_V::step_V]
    df = df[df["V"].isin(selected_V)]

    df_per_V = list()
    unique_V = df["V"].unique()
    unique_V.sort()
    for v in unique_V:
        subset_df = df[df["V"] == v]
        assert len(subset_df)>1, v
        n_points_per_V[v] = len(subset_df)
        unique_Q = subset_df["Q"].unique()
        unique_Q.sort()
        selected_Q = unique_Q[0::step_Q]
        subset_df = subset_df[subset_df["Q"].isin(selected_Q)]

        unique_P = subset_df["P"].unique()
        unique_P.sort()
        selected_P = unique_P[0::step_P]
        subset_df = subset_df[subset_df["P"].isin(selected_P)]

        #print(v,len(selected_Q),len(selected_Q),len(subset_df))

        df_per_V.append(subset_df)
    df_per_V = pd.concat(df_per_V).reset_index(drop=True)
    return df_per_V

def load_dataset(dataset="OP_sin", return_type="dataframe",select_features="all",test_split="none", random_state=None,
                 grid_resolution=1000, specific_points=None):
    assert dataset in ["OP_mul","OP_sin","OP_sin_more","specific_points"], f"Invalid dataset name {dataset}"
    if "random" in test_split:
        assert random_state is not None, "Random state must be set"
    print(f"Loading dataset {dataset} with test split {test_split}, specific point {specific_points}")
    features = get_dataset_feature_list(dataset=dataset, specific_points=specific_points)

    df = add_pu_units(raw_data_to_df(dataset, specific_points=specific_points))
    X,y = dataset_df_to_Xy(df, features)

    if test_split == "none":
        if return_type == "dataframe":
            return df
        elif return_type == "X_y":
            return X,y
    elif "random" in test_split:
        train_ratio = float(test_split.split("_")[-1])
        #print(f"Randomly splitting data into train and test with ratio {train_ratio}")
        # numpy random seed object
        rng = np.random.default_rng(random_state)
        # randomly split the data into train and test with a seed
        train_idx = rng.choice(X.shape[0], int(train_ratio*X.shape[0]), replace=False, )
        X_train, y_train = X[train_idx], y[train_idx]
        df["is_test"] = True
        df.loc[train_idx,"is_test"] = False
        if return_type == "X_y":
            return X_train, X, y_train, y
        elif return_type == "dataframe":
            return df

def load_sklearn_split(dataset="OP_sin", test_size=0.5, random_state=42):
    #imbalance_strategy = "over_sample"
    X,y = load_dataset(dataset=dataset, return_type="X_y")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def populate_df_with_preds_from_threshold(args, df_test_preds, df_thresh):

    args_dict = args.to_dict().copy()
    
    # make sure every key in args_dict is in df_thresh
    for key in args_dict.keys():
        if key not in df_thresh.columns:
            print(f"Key {key} not in df_thresh")
            df_thresh[key] = args_dict[key]

    df_thresh = subset_df(df_thresh, args_dict)
    
    df_thresh = df_thresh[df_thresh["precision_test"] == df_thresh["precision_test"].min()].reset_index(drop=True)
    t1 = df_thresh["threshold_t1"][0]
    t2 = df_thresh["threshold_t2"][0]

    df_test_preds["pred_cls"] = df_test_preds["pred_prob"].apply(lambda x: str_pred_from_prob(x,t1,t2))
    
    def evaluate_row(row):
        if row["pred_cls"] == "stable":
            if row["S_flip"] == 1:
                return "True Positive"
            else:
                return "False Positive"
        elif row["pred_cls"] == "unstable":
            if row["S_flip"] == 0:
                return "True Negative"
            else:
                return "False Negative"
        else:
            return "Rejected"
        
    df_test_preds["status"] = df_test_preds.apply(lambda x: evaluate_row(x), axis=1)
    return df_test_preds

def add_pu_units(df):
    df.loc[:,"V (p. u.)"] = df["V"].apply(lambda x: x/110)

    if "P" in df.columns and "Q" in df.columns:
        df.loc[:,"P (p. u.)"] = df["P"].apply(lambda x: x/3000)
        df.loc[:,"Q (p. u.)"] = df["Q"].apply(lambda x: x/3000)
        feats = ["V","P","Q"]
    elif "P1" in df.columns and "Q1" in df.columns:
        df.loc[:,"P1 (p. u.)"] = df["P1"].apply(lambda x: x/3000)
        df.loc[:,"Q1 (p. u.)"] = df["Q1"].apply(lambda x: x/3000)
        df.loc[:,"P2 (p. u.)"] = df["P2"].apply(lambda x: x/3000)
        df.loc[:,"Q2 (p. u.)"] = df["Q2"].apply(lambda x: x/3000)
        feats = ["V","P1","Q1","P2","Q2"]
    
    # for feat in feats:
    #     print(feat, df[f'{feat}'].min(), df[f'{feat}'].max())
    #     print(feat, df[f'{feat} (p. u.)'].min(), df[f'{feat} (p. u.)'].max())
    return df

def test_eval_visualisation(args, df_test_preds=None, df_thresh=None, **kwargs):
    path_name = path_name_from_args(args, eval=True)

    Vs_to_vis = df_test_preds["V"].unique()[:1]

    for selected_V in Vs_to_vis:
        df_test_preds_selected = df_test_preds[df_test_preds["V"] == selected_V]
        
        print("Selected V value in test preds:", selected_V)

        canvas_width_point = 200*1.4
        canvas_width_tile = 200*0.3
        n_P = len(df_test_preds_selected["P"].unique())
        point_size = canvas_width_point / n_P / 2
        tile_size = canvas_width_tile / n_P / 2
        print(f"Point size: {point_size}, tile size: {tile_size}")

        plot = (
                ggplot(df_test_preds_selected)
                + geom_tile(aes("P","Q",fill="pred_prob"), size=tile_size)
                + scale_fill_continuous(limits=[0,1], cmap_name="gray", name="Estimated p")
                + theme_light()
                + theme(figure_size=(6,4))
            )
        plot.save(f"slices/slices-{path_name}-conf_V{selected_V}.png", dpi=100)

        plot = (
                ggplot(df_test_preds_selected)
                + geom_raster(aes("P (p. u.)","Q (p. u.)",fill="pred_prob"))
                + theme_light()
                + theme(figure_size=(3,3),legend_position="none")
            )
        df_test_preds_selected.to_csv(f"slices/conf_slice_{path_name}.csv")
        plot.save(f"slices/slices-{path_name}-conf_V{selected_V}_pu.png", dpi=300)

        for threshold_mode in ["20rejection_or_highprecision"]:
            df_test_preds_selected = populate_df_with_preds_from_threshold(args, df_test_preds_selected, df_thresh=df_thresh)

            status_to_color = {
                "True Positive": "green",
                "False Positive": "red",
                "True Negative": "blue",
                "False Negative": "orange",
                "Rejected": "pink"
            }
            
            plot = (
                ggplot(df_test_preds)
                + geom_point(df_test_preds_selected, aes("P","Q",color="status"), size=point_size, alpha=1, stroke=0)
                + scale_color_manual(values=status_to_color, name=" ")
                + theme_light()
                + theme(figure_size=(6,4))
            )
            #plot.save(f"slices/slices-{path_name}-cls_{threshold_mode}_V{selected_V}.pdf")
            plot.save(f"slices/slices-{path_name}-cls_{threshold_mode}_V{selected_V}.png", dpi=300)

            plot = (
                ggplot(df_test_preds)
                + geom_point(df_test_preds_selected, aes("P (p. u.)","Q (p. u.)",color="status"), size=point_size, alpha=1, stroke=0)
                + scale_color_manual(values=status_to_color, name=" ")
                + theme_light()
                + theme(figure_size=(6,4))
            )
            df_test_preds_selected.to_csv(f"slices/cls_slice_{path_name}.csv")
            #plot.save(f"slices/slices-{path_name}-cls_{threshold_mode}_V{selected_V}_pu.pdf")
            plot.save(f"slices/slices-{path_name}-cls_{threshold_mode}_V{selected_V}_pu.png", dpi=300)

def entropy_from_prob(prob):
    h = - prob * np.log2(prob) - (1-prob) * np.log2(1-prob)
    return h

def is_rejected(prob, t1, t2):
    return (prob > t1) & (prob < t2)

def is_stable(prob, t1, t2):
    return prob >= t2

def is_unstable(prob, t1, t2):
    return prob <= t1

def str_pred_from_prob(x,t1,t2):
    return "stable" if is_stable(x,t1,t2) else "rejected" if is_rejected(x, t1, t2) else "unstable"

def precision_per_rejection_threshold(pred_prob, gt, ax, label, reject_rate=0.2, pos_label=1):
    precisions, recalls, valid_thresholds = precision_recall_curve(gt, pred_prob)
    valid_thresholds = np.append(valid_thresholds, [1], axis=0)

    highrecall = np.max(recalls)
    highrecall_idx = np.where(recalls==highrecall)[0]

    highrecall_idx = highrecall_idx.max()
    threshold_highrecall = valid_thresholds[highrecall_idx]
    
    total_preds = len(pred_prob)
    retain_rate = 1-reject_rate 
    retain_count = int(retain_rate * total_preds)
    sorted_pred_prob = np.sort(pred_prob)

    T1 = threshold_highrecall

    below_T1 = np.sum(is_unstable(pred_prob, T1, None))
    retain_above_T2 = retain_count - below_T1
    
    T2_index = total_preds - retain_above_T2 
    T2_index = min(total_preds-1, T2_index) 
    T2 = sorted_pred_prob[T2_index]


    print(f"T1: {T1}, T2: {T2}")

    print(np.mean(is_rejected(pred_prob, T1, T2)))

    assert T1 <= T2

    res_dict = {
        "t1": T1,
        "t2": T2,
        "model_name": label
    }
    return res_dict

def precision_per_threshold_efficient(pred_prob, gt, ax, label, thresholds=np.array(range(1000+1))/1000, pos_label=1,
                                      plot_proportionvsrecall=False, plot_proportionvstnsfns=False):

    precisions, recalls, valid_thresholds = precision_recall_curve(gt, pred_prob)

    valid_thresholds = np.append(valid_thresholds, [1], axis=0)

    highprecision = np.max(precisions)
    highprecision_idx = np.argmax(precisions).min()

    recall_highprecision = recalls[highprecision_idx]
    threshold_highprecision = valid_thresholds[highprecision_idx]

    highrecall = np.max(recalls)
    highrecall_idx = np.where(recalls==highrecall)[0]
    #print(highrecall_idx)
    highrecall_idx = highrecall_idx.max()

    precision_highrecall = precisions[highrecall_idx]
    threshold_highrecall = valid_thresholds[highrecall_idx]


    res_dict = {
        "highprecision": highprecision,
        "recall_highprecision": recall_highprecision,
        "threshold_highprecision": threshold_highprecision,
        "highrecall": highrecall,
        "precision_highrecall": precision_highrecall,
        "threshold_highrecall": threshold_highrecall,
        "model_name": label
    }
    
    return res_dict
