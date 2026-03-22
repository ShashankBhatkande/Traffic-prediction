import optuna
from main import main  # your existing main function

def objective(trial):
    # 🔥 Hyperparameters to tune
    args = {}

    args["model_name"] = "TGCN"
    args["data"] = "losloop"
    args["settings"] = "supervised"

    # Tunable parameters
    args["learning_rate"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    args["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    args["seq_len"] = trial.suggest_int("seq_len", 8, 36)
    args["pre_len"] = trial.suggest_int("pre_len", 1, 3)
    args["batch_size"] = trial.suggest_categorical("batch_size", [32, 64])
    args["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    args["max_epochs"] = 20  # keep small for tuning

    # Convert dict to argparse-like object
    class Args:
        pass

    arg_obj = Args()
    for k, v in args.items():
        setattr(arg_obj, k, v)

    # 🔥 Run training
    results = main(arg_obj)

    # IMPORTANT: return RMSE (or val_loss)
    return results["RMSE"]


# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best parameters:", study.best_params)