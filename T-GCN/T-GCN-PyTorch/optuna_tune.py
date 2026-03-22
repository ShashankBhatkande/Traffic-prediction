import optuna
from main import main  # your existing main function
import argparse

def objective(trial):
    # 🔥 Hyperparameters to tune
    args = argparse.Namespace()

    args.model_name = "TGCN"
    args.data = "losloop"
    args.settings = "supervised"
    args.optuna = True

    # Tunable parameters
    args.learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    args.seq_len = trial.suggest_int("seq_len", 8, 36)
    args.pre_len = trial.suggest_int("pre_len", 1, 3)
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64])
    args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    args.normalize = True
    args.split_ratio = 0.8

    # ⚡ FAST training (tuning phase)
    args.max_epochs = 20
    args.accelerator = "cpu"
    args.devices = 1

    # Convert dict to argparse-like object
    args.log_path = None
    args.send_email = False

    results = main(args)

    return results["RMSE"]


# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20, n_jobs=4)

print("Best parameters:", study.best_params)