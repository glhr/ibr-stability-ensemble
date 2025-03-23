import numpy as np
import dill

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support

import data_utils as du
import pandas as pd


NICE_DATASET_NAMES = {
    "OP_sin": "Single dataset",
    "OP_mul": "Multi dataset"
}

class EnsembleMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, ensemble_models=[]):
        
        self.ensemble_models = ensemble_models
        self.ensemble_n = len(ensemble_models)
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        #X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        # Return the classifier
        for n in range(self.ensemble_n):
            #print(f"Fitting {n}")
            self.ensemble_models[n].fit(X,y)
        self.is_fitted_ = True
        self.loss_curve_ = self.ensemble_models[-1].loss_curve_
        return self
        
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        preds = []
        for n in range(self.ensemble_n):
            preds.append(self.ensemble_models[n].predict_proba(X))
        preds = np.stack(preds)
        #print(preds.shape)

        preds_mean = preds.mean(axis=0)
        #print(preds_mean.shape)

        #closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        #return self.y_[closest]
        return np.argmax(preds_mean,axis=-1)

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        preds = []
        for n in range(self.ensemble_n):
            preds.append(self.ensemble_models[n].predict_proba(X))
        preds = np.stack(preds)
        #print(preds.shape)

        preds_mean = preds.mean(axis=0)
        #print(preds_mean.shape)
            
        #closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return preds_mean

def load_model(pkl_path):
    model_path = pkl_path + "_mlp.pkl"
    with open(model_path, "rb") as f:
        model = dill.load(f)
    return model

def train_sklearn_deterministic(random_state,X_train,Y_train,hidden_layer_sizes,activation="sigmoid",smoke_test=False,
                                         alpha=0, solver="adam",pkl_path="", ensemble_n=1,**kwargs):


    import numpy as np
    
    import random
    import os

    if activation == "sigmoid":
        activation = "logistic"

    mlp_path = pkl_path + "_mlp.pkl"
    
    # if the model already exists, return
    #if os.path.exists(mlp_path):
    #    return mlp_path

    LEARNING_RATE = 0.1
    TOL = 1e-4
    MAX_ITER = 1000

    if ensemble_n ==1 :
        mlp = MLPClassifier(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes,
                            solver=solver,activation=activation,
                            learning_rate_init=LEARNING_RATE, learning_rate="adaptive",verbose=False,
                            random_state=random_state,max_iter=100 if smoke_test else MAX_ITER,
                            tol=TOL)
    
    elif ensemble_n > 1:
        ensemble_models = []
        for n in range(ensemble_n):
            random_state_ensemble = random_state if n == 0 else random_state * 100 + n
            ensemble_models.append(
                MLPClassifier(
                    alpha=alpha, hidden_layer_sizes=hidden_layer_sizes,
                    solver=solver,activation=activation,
                    learning_rate_init=LEARNING_RATE, learning_rate="adaptive",verbose=False,
                    tol=TOL,
                    random_state=random_state_ensemble,max_iter=100 if smoke_test else MAX_ITER
                )
            )
        mlp = EnsembleMLPClassifier(ensemble_models=ensemble_models)
    mlp.fit(X_train, Y_train)
    losses = mlp.loss_curve_

    # save the model and losses
    with open(mlp_path, "wb") as f:
        dill.dump(mlp, f)
        
    return mlp_path


def convert_sklearn_ensemble_to_hummingbird(ensemble_model, mode="GPU",X_train=None,Y_train=None,**kwargs):
    from sklearn.ensemble import StackingClassifier
    from sklearn.preprocessing import FunctionTransformer
    from hummingbird.ml import convert, load
    from sklearn.linear_model import LogisticRegression
    
    ensemble_members = ensemble_model.ensemble_models
    ensemble_n = ensemble_model.ensemble_n

    ensemble_members_hb = []
    for n,ensemble_member in enumerate(ensemble_members):
        ensemble_members_hb.append(
            (str(n),ensemble_member
            )
        )

    final_classifier = LogisticRegression(random_state=42)

    mlp = StackingClassifier(
            estimators=ensemble_members_hb, final_estimator=final_classifier, passthrough=False,
            stack_method="predict_proba", cv="prefit"
        )
    #mlp.fit(X_train,Y_train)
    mlp.final_estimator_ = "identity"
    mlp.estimators_ = ensemble_members
    mlp.stack_method_ = ["predict_proba" for _ in range(ensemble_n)]
    predict_func = lambda m,x: m.model.forward(x).reshape(x.shape[0],-1,2).mean(axis=1).cpu().numpy()

    if mode == "GPU":
        model = convert(mlp, 'pytorch')
        # Run predictions on GPU
        model.to('cuda')
    else:
        model = convert(mlp, 'pytorch')

    return model, predict_func

def convert_sklearn_mlp_to_hummingbird(mlp, mode="GPU",X_train=None,Y_train=None,**kwargs):
    from hummingbird.ml import convert, load

    predict_func = lambda m,x: m.predict_proba(x)

    #mlp.fit(X_train,Y_train)

    if mode == "GPU":
        model = convert(mlp, 'pytorch')
        # Run predictions on GPU
        model.to('cuda')
    else:
        model = convert(mlp, 'pytorch')
    return model, predict_func



def eval_failed(args):
    path_template = du.path_name_from_args(args)
    print("Eval failed for", path_template)

def eval_model(args, seed_val=0, threshold_mode="highprecision", val_ratio=0.2, csv_preds=None):
    path_template_id = du.path_name_from_args(args,eval=False)
    assert csv_preds == f"preds_csv/df_test-{path_template_id}.csv"
    df_dataset = pd.read_csv(csv_preds)
    if "S_flip" not in df_dataset.columns:
        return eval_failed(args)
    
    model_name = args.model_name
    
    #print(args.to_dict())
    #print(df_test["is_test"])
    #try:
    #    conf_metric = SCORING_PER_MODEL[model_name]
    #except Exception as e:
    #    conf_metric = SCORING_PER_MODEL["_".join(model_name.split("_")[:-1])]
    df_evalset = df_dataset[df_dataset["is_test"]]
    if not len(df_evalset) > 0:
        return eval_failed(args)
    #df_evalset["model_name"] = model_name

    # split into val and test by splitting the evalset randomly
    # create numpy rng
    if val_ratio < 1:
        rng = np.random.default_rng(seed=seed_val)
        val_idx = rng.choice(df_evalset.index, np.round(len(df_evalset)*val_ratio).astype(int), replace=False)
        df_val = df_evalset.loc[val_idx]
        gt_val = df_val["S_flip"]
        prob_val = df_val["pred_prob"]
        df_test = df_evalset.drop(val_idx)
    else:
        df_test = df_evalset
        prob_val = df_evalset["pred_prob"]
        gt_val = df_evalset["S_flip"]

    if args.dataset == args.dataset_eval:
        df_separate_test = df_test.copy()
    else:
        path_name_ood = du.path_name_from_args(args,eval=True)
        df_separate_test = pd.read_csv(f"preds_csv/df_test-{path_name_ood}.csv")
        #df_separate_test["is_test"] = df_separate_test["P"].apply(lambda x: 1320 > abs(x) > 750)
        #df_separate_test = df_separate_test[df_separate_test["is_test"]]

    gt_test = df_separate_test["S_flip"]
    prob_test = df_separate_test["pred_prob"]

    # plt.hist(prob_test, bins=100)
    # plt.title(f"Test probs")
    # plt.show()
    
    if "rejection_or_highprecision" in threshold_mode:
        rejection_rate = float(threshold_mode.split("rejection")[0])/100
        print(f"Using rejection rate", rejection_rate)
        res_dict_val_20rejection = du.precision_per_rejection_threshold(prob_val, gt_val, ax=None, label=model_name, reject_rate=rejection_rate)
        res_dict_val_highprecision = du.precision_per_threshold_efficient(prob_val, gt_val, ax=None, label=model_name, plot_proportionvsrecall=False, plot_proportionvstnsfns=False)

        threshold_t1_20rejection = res_dict_val_20rejection["t1"]
        threshold_t2_20rejection = res_dict_val_20rejection["t2"]
        
        threshold_t1_highprecision = min(res_dict_val_highprecision["threshold_highprecision"], res_dict_val_highprecision["threshold_highrecall"])
        threshold_t2_highprecision = max(res_dict_val_highprecision["threshold_highprecision"], res_dict_val_highprecision["threshold_highrecall"])

        assert threshold_t1_highprecision == threshold_t1_20rejection

        mask_rejection_highprecision = du.is_rejected(prob_test, threshold_t1_highprecision, threshold_t2_highprecision)
        mask_rejection_20rejection = du.is_rejected(prob_test, threshold_t1_20rejection, threshold_t2_20rejection)
        rejection_rate_highprecision = np.mean(mask_rejection_highprecision.astype(int))
        rejection_rate_20rejection = np.mean(mask_rejection_20rejection.astype(int))

        #print(f"---")
        #print(f"highprecision threshs: {threshold_t1_highprecision:.2f}, {threshold_t2_highprecision:.2f} | rejection rate: {rejection_rate_highprecision:.2f}")
        #print(f"rejection threshs: {threshold_t1_20rejection:.2f}, {threshold_t2_20rejection:.2f} | rejection rate: {rejection_rate_20rejection:.2f}")

        threshold_t2 = max(threshold_t2_20rejection, threshold_t2_highprecision)
        threshold_t1 = threshold_t1_highprecision
    else:
        raise NotImplementedError

    cls_test = prob_test >= threshold_t1
    mask_rejection = du.is_rejected(prob_test, threshold_t1, threshold_t2)
    
    precision_test_1, recall_test_1 = precision_recall_fscore_support(gt_test, cls_test, average="binary", sample_weight=(~mask_rejection).astype(int))[:2]
    

    rejection_rate = np.mean(mask_rejection.astype(int))

    # true_negatives = np.sum((gt_test[~mask_rejection] == 0) & (cls_test[~mask_rejection] == 0))
    false_positives = (gt_test == 0) & (cls_test == 1)
    if np.sum((false_positives) & (~mask_rejection)) > 1:
        df_separate_test["FP"] = (false_positives) & (~mask_rejection)
        df_separate_test_FP = df_separate_test.loc[df_separate_test["FP"]]
        try:
            df_separate_test_FP.to_csv(f"FP_csv/df_test-{path_name_ood}-seed_val{seed_val}.csv")
        except:
            df_separate_test_FP.to_csv(f"FP_csv/df_test-{path_template_id}-seed_val{seed_val}.csv")

    # false_positive_rate = (false_positives) / (false_positives+true_negatives)

    # true_positives = np.sum((gt_test[~mask_rejection] == 1) & (cls_test[~mask_rejection] == 1))
    # false_negatives = np.sum((gt_test[~mask_rejection] == 1) & (cls_test[~mask_rejection] == 0))
    # false_negative_rate = (false_negatives) / (false_negatives+true_positives)
    

    res_dict = {
        "precision_test": precision_test_1,
        "recall_test": recall_test_1,
        "threshold_t1": threshold_t1,
        "threshold_t2": threshold_t2,
        "rejection_rate": rejection_rate,
        #"false_positive_rate": false_positive_rate,
        #"false_negative_rate": false_negative_rate,
    }

    #print(f"Rejection rate: {rejection_rate:.2f} - precision {precision_test_1}")

    if precision_test_1 < 1:
        print(args.to_dict())

    
    res_dict.update(args.to_dict())
    return res_dict

def train_model(model_name,random_state,
                   X_train,Y_train,
                   hidden_layer_sizes=(50,50),
                   activation="relu",
                   smoke_test=False,
                   **kwargs
                  ):
    if model_name == "sklearn_mlp":
        func = train_sklearn_deterministic
        kwargs = {"alpha": 0, **kwargs}
    elif "sklearn_mlp_ensemble" in model_name:
        func = train_sklearn_deterministic
        ensemble_n = int(model_name.split("_")[-2])
        alpha = float(model_name.split("_")[-1])
        assert ensemble_n > 1
        kwargs = {"ensemble_n": ensemble_n, "alpha": alpha, **kwargs}
        kwargs = {**kwargs}
    else:
        print(f"Model {model_name} not implemented")
        raise NotImplementedError

    return func(random_state=random_state,X_train=X_train,Y_train=Y_train,
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                smoke_test=smoke_test,
                **kwargs)