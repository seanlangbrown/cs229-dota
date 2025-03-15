from data_loader import create_batch_dataloader, train_evel_test_split, get_features_in_out, get_embedding_dimensions, remove_embedding_columns
from models import EmbeddingFF
import torch
import time
import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from cs229_utils import snake_case
import json
from train_ff import get_available_gpu, calc_accuracy, get_optimizer, should_stop_early, plotEpochStats


config = {
    "model_name": "FFWithEmbeddings",
    "embedding": True,
    "embeddings": {
        "heroEmbeddings": True,
        "abilityEmbeddings": False,
    },
    "one_hot": False,
    "batches": 100, # can be used to limit the number of batches per epoch for testing
    "batch_size": 32,
    "epochs": 15,
    # "batches": 10, # can be used to limit the number of batches per epoch for testing
    # "batch_size": 32,
    # "epochs": 3,
    "optimizer": "Adam", # "SGD"
    "optimizer_params_adam": {
        "lr": 0.001,
    },
    "learning_rate": 0.001,
    "save_i_epochs": 3,
    "skip_all_data_errors": False,
    "early_stopping_patience": 3,
    "allow_no_tower_kill_matches": False, # remove matches with no tower kills from the sample to keep it balanced
    "dropout": 0.2,
}


def train():
    print(f"training {config['model_name']}")

    device, isCPU = get_available_gpu()

    train_data, eval_data, test_data = train_evel_test_split(allowNoMatches=config.get("allow_no_tower_kill_matches", False))

    print(f"training with {len(train_data)} training matches and {len(eval_data)} evaluation matches")

    should_shuffle = config["batches"] is not None # if we're not training on all possible batches, then we need to randomize further
    train_data_loader = create_batch_dataloader(train_data, batch_size=config["batch_size"], shuffle=should_shuffle, config=config, training_cpu=isCPU) # shuffling is not usually needed because of how train_sample is picked - it will be in random order already
    eval_data_loader = create_batch_dataloader(eval_data, batch_size=config["batch_size"], shuffle=False, config=config, training_cpu=isCPU)

    feat_in, feat_out = get_features_in_out(oneHot=config["one_hot"], embedding=config["embedding"], embeddingConfig=config["embeddings"])

    

    print(f"training with {len(feat_in)} features and {len(feat_out)} labels")
    print(f"training for {config['epochs']} epochs of up to {config['batches']} batches of batch size {config['batch_size']}")

    model = EmbeddingFF(num_features=len(remove_embedding_columns(feat_in)), num_labels=len(feat_out), categorical_features_info=get_embedding_dimensions(embeddingConfig=config["embeddings"]), config=config)

    model.to(device=device) # move model to cpu/gpu
    # TODO: need to move input data also

    # criterion = torch.nn.CrossEntropyLoss()
    # binary_classification_loss = torch.nn.BCELoss()
    criterion = torch.nn.BCEWithLogitsLoss()  # Combines sigmoid and BCE

    OptimizerType, optimizer_params = get_optimizer(config)

    optimizer = OptimizerType(model.parameters(), **optimizer_params)


    # Train stats
    accuracy_log = []
    loss_log = []
    batch_sizes = []
    total_matches = 0

    # Eval stats
    accuracy_eval_log = []
    loss_eval_log = []
    batch_eval_sizes = []
    total_eval_matches = 0

    should_stop_now = should_stop_early(config["early_stopping_patience"])

    num_epochs = config["epochs"]
    for epoch in range(num_epochs):
        startEpochTime = time.time()
        np.random.seed() # reset seed   https://github.com/pytorch/pytorch/issues/5059  data loader returns the same values
        epoch_sum_loss = 0.0
        epoch_sum_accuracy = 0.0
        train_batches = 0
        
        for batch_i, (batch_x_obj, batch_y_obj) in enumerate(train_data_loader):
            if config["batches"] is not None and batch_i >= config["batches"]:
                break

            if not batch_x_obj["valid"]:
                continue

            train_batches += 1

            batch_sizes.append(batch_x_obj["tensor"].shape[0])
            total_matches += batch_x_obj["count_matches"]

            batch_x = batch_x_obj["tensor"].to(device)
            batch_y = batch_y_obj["tensor"].to(device)
            categorical_batch_x = batch_x_obj["categorical_features"]
            categorical_batch_x = {k: v.to(device) for k, v in categorical_batch_x.items()}

            # Forward pass
            output = model(batch_x, categorical_batch_x)
            loss = criterion(output, batch_y)
            batch_accuracy = calc_accuracy(output, batch_y)
            epoch_sum_accuracy += batch_accuracy
            
            
            # Backward pass and optimize
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights

            epoch_sum_loss += loss.item()

            # Print batch statistics?

        # save epoch stats for plotting
        accuracy_log.append(epoch_sum_accuracy / train_batches)
        loss_log.append(epoch_sum_loss / train_batches)

        epoch_loss = epoch_sum_loss/train_batches
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_sum_loss / train_batches:.4f}')

        with torch.no_grad():
            epoch_sum_eval_loss = 0.0
            epoch_sum_eval_accuracy = 0.0
            eval_batches = 0

            for batch_i, (batch_x_obj, batch_y_obj) in enumerate(eval_data_loader):
                if batch_i >= config["batches"]:
                    break

                if not batch_x_obj["valid"]:
                    continue

                eval_batches+=1

                batch_eval_sizes.append(batch_x_obj["tensor"].shape[0])
                total_eval_matches += batch_x_obj["count_matches"]

                batch_x = batch_x_obj["tensor"].to(device)
                batch_y = batch_y_obj["tensor"].to(device)
                categorical_batch_x = batch_x_obj["categorical_features"]
                categorical_batch_x = {k: v.to(device) for k, v in categorical_batch_x.items()}

                # Predict
                output = model(batch_x, categorical_batch_x)
                loss = criterion(output, batch_y)
                batch_accuracy = calc_accuracy(output, batch_y)
                epoch_sum_eval_accuracy += batch_accuracy

                epoch_sum_eval_loss += loss.item()

            # save epoch stats for plotting
            epoch_eval_loss = epoch_sum_eval_loss / eval_batches
            accuracy_eval_log.append(epoch_sum_eval_accuracy / eval_batches)
            loss_eval_log.append(epoch_eval_loss)

            # Print epoch statistics
            print(f'Epoch {epoch+1}/{num_epochs}, Eval Loss: {epoch_eval_loss:.4f}, Eval Accuracy: {epoch_sum_eval_accuracy / eval_batches:.4f}')

            

        print("Epoch took: ",time.time()-startEpochTime)
        sys.stdout.flush()

        stop_now = should_stop_now(epoch_eval_loss)

        if config["save_i_epochs"] or stop_now:
            i = config["save_i_epochs"]
            if (epoch % i) == i-1:
                    torch.save(model.state_dict(), f"./training/{snake_case(config['model_name'])}_" + str(epoch) + ".model")

        if stop_now:
            print("stopping early")
            break

    # Plot stats
    plotEpochStats((loss_log, loss_eval_log), "Binary Classification Loss", ["Train", "Validation"], config)
    plotEpochStats((accuracy_log, accuracy_eval_log), "Accuracy", ["Train", "Validation"], config)

    print(f"trained on {total_matches} matches with {np.sum(batch_sizes)} rows")
    print(f"evaluated on {total_eval_matches} with {np.sum(batch_eval_sizes)} rows")

    results_summary = {
        "config": config,
        "total_matches_train": total_matches,
        "total_rows_train": batch_sizes,
        "total_matches_eval": total_eval_matches,
        "total_rows_eval": batch_eval_sizes,
        "total_matches_test": 0,
        "total_rows_test": 0,
        "final_loss_train": loss_log[-1],
        "final_loss_eval": loss_eval_log[-1],
        "final_loss_test":0,
        "final_acc_train": accuracy_log[-1],
        "final_acc_eval": accuracy_eval_log[-1],
        "final_acc_test": 0,
        "loss_log_train": loss_log,
        "loss_log_eval": loss_eval_log,
        "loss_log_test": [],
        "acc_log_train": accuracy_log,
        "acc_log_eval": accuracy_eval_log,
        "acc_log_test": [],
    }

    with open(f"./training/{snake_case(config['model_name'])}_results.json", 'w') as f:
        json.dump(results_summary, f, indent=4)


if __name__ == "__main__":

    train()