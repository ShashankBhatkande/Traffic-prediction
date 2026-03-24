import optuna
from main import main
import argparse


def objective(trial):
    args = argparse.Namespace()

    args.model_name = "TGCN"
    args.data = "losloop"
    args.settings = "supervised"
    args.optuna = True

    # 🔥 Tunable
    args.learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
    args.seq_len = trial.suggest_int("seq_len", 8, 24)
    args.pre_len = trial.suggest_int("pre_len", 1, 3)
    args.batch_size = trial.suggest_categorical("batch_size", [16, 32])
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    args.rnn_type = "none"  # disable heavy GRU in auto-tuning by default

    # 🔥 IMPORTANT
    args.normalize = True
    args.split_ratio = 0.8

    # 🔥 FAST tuning
    args.max_epochs = 8
    args.accelerator = "cpu"
    args.devices = 1

    args.log_path = None

    results = main(args)
    return results["RMSE"]


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20, n_jobs=4)

print("Best params:", study.best_params)