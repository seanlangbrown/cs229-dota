from data_loader import create_batch_dataloader, train_evel_test_split, get_features_in_out
from models import SimpleFF
import torch
import time
import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from cs229_utils import snake_case
import json
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


config = {
    "model_name": "SimpleFFTest",
    "embedding": False,
    "embeddings": {
        "abilityEmbedding": False,
        "heroEmbedding": False,
    },
    "one_hot": False,
    "batches": 100, # can be used to limit the number of batches per epoch for testing
    "batch_size": 32,
    "epochs": 15,
    "optimizer": "Adam", # "SGD"
    "optimizer_params_adam": {
        "lr": 0.001,
    },
    "learning_rate": 0.001,
    "save_i_epochs": 3,
    "skip_all_data_errors": False,
    "early_stopping_patience": 3,
    "allow_no_tower_kill_matches": False, # remove matches with no tower kills from the sample to keep it balanced
}


def should_stop_early(patience):
    state = {
        "best_val_loss": None,
        "patience_counter": 0,
    }
    
    def stop_now(val_loss):
        if state["best_val_loss"] is None: # type: ignore
            state["best_val_loss"] = val_loss
            return False

        if val_loss < state["best_val_loss"]: # type: ignore
            state["best_val_loss"] = val_loss
            state["patience_counter"] = 0
            return False
        else:
            state["patience_counter"] += 1 # type: ignore
            if state["patience_counter"] >= patience:
                return True
    return stop_now

def plotEpochStats(values, name, legends, config, show=False):
    plt.clf()

    # Use the newer colormaps.get_cmap() method
    cmap = mpl.colormaps.get_cmap('Set2')
    colors = cmap(np.linspace(0, 1, len(values)))
    
    # Plot each vector with a distinct color
    for i, vec in enumerate(values):
        plt.plot(vec, color=colors[i], linewidth=2, marker='o', markersize=4, alpha=0.8)
    
    plt.ylabel(name, fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.title(f'{name} vs Epoch', fontsize=14)
    
    plt.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5),
               frameon=True, fancybox=True, shadow=False, fontsize=10)
    
    plt.tight_layout(pad=2.0)

    plt.savefig(f"./training/{snake_case(config['model_name'])}_{snake_case(name)}.png", bbox_inches='tight', dpi=300)

    if show:
        plt.show()

def get_available_gpu():
    # should this be "cuda:0"?
    name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{name} available")
    return torch.device(name), name == "cpu"  # use gpu if available, else cpu

def get_optimizer(config):
    if config["optimizer"] == "Adam":
        OptimizerType = torch.optim.Adam
        optimizer_params = config["optimizer_params_adam"]
    elif config["optimizer"] == "SGD":
        OptimizerType = torch.optim.SGD
        optimizer_params = config["optimizer_params_sgd"]

    return OptimizerType, optimizer_params # type: ignore

def calc_accuracy(output, y):
    predictions = torch.sigmoid(output) >= 0.5  # this is multi-label classification, so both can be t/f, (not multi-class)

    # print(predictions.shape)
    # print(y.shape)

    accuracy_vec = (predictions == y).cpu().numpy().reshape(-1).astype(np.float32)
    overall_accuracy = accuracy_vec.mean()
    return overall_accuracy


def train():
    print(f"training {config['model_name']}")

    device, isCPU = get_available_gpu()

    train_data, eval_data, test_data = train_evel_test_split(allowNoMatches=config.get("allow_no_tower_kill_matches", False))

    print(f"training with {len(train_data)} training matches and {len(eval_data)} evaluation matches")

    should_shuffle = config["batches"] is not None # if we're not training on all possible batches, then we need to randomize further
    train_data_loader = create_batch_dataloader(train_data, batch_size=config["batch_size"], shuffle=should_shuffle, config=config, training_cpu=isCPU) # shuffling is not usually needed because of how train_sample is picked - it will be in random order already
    eval_data_loader = create_batch_dataloader(eval_data, batch_size=config["batch_size"], shuffle=False, config=config, training_cpu=isCPU)
    test_data_loader = create_batch_dataloader(test_data, batch_size=config["batch_size"], shuffle=False, config=config, training_cpu=isCPU)


    feat_in, feat_out = get_features_in_out(oneHot=config["one_hot"], embedding=config["embedding"])

    print(f"training with {len(feat_in)} features and {len(feat_out)} labels")
    print(f"training for {config['epochs']} epochs of up to {config['batches']} batches of batch size {config['batch_size']}")

    model = SimpleFF(num_features=len(feat_in), num_labels=len(feat_out))

    model.to(device=device) # move model to cpu/gpu
    # TODO: need to move input data also


    # criterion = torch.nn.CrossEntropyLoss()
    # binary_classification_loss = torch.nn.BCELoss()
    criterion = torch.nn.BCEWithLogitsLoss()  # Combines sigmoid and BCE

    OptimizerType, optimizer_params = get_optimizer(config)

    optimizer = OptimizerType(model.parameters(), **optimizer_params)


    if False: #Load and Test only
        test_model(model, test_data_loader, device, criterion)
        return

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

            # Forward pass
            output = model(batch_x)
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

                # Predict
                output = model(batch_x)
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


def test_model(model, test_data_loader, device, criterion, file="./training/simple_f_f_test_14.model"):
    model.load_state_dict(torch.load(file))
    # Set to evaluation mode
    model.eval()


    y_true_1 = []
    y_pred_1 = []
    y_true_2 = []
    y_pred_2 = []

    # Eval stats
    accuracy_test_log = []
    loss_test_log = []
    batch_test_sizes = []
    total_test_matches = 0

    with torch.no_grad():
        epoch_sum_train_loss = 0.0
        epoch_sum_train_accuracy = 0.0
        train_batches = 0

        for batch_i, (batch_x_obj, batch_y_obj) in enumerate(test_data_loader):
            if batch_i >= config["batches"]:
                break

            if not batch_x_obj["valid"]:
                continue

            train_batches+=1

            batch_x = batch_x_obj["tensor"].to(device)
            batch_y = batch_y_obj["tensor"].to(device)

            total_test_matches += batch_x.shape[0]

            # Predict
            output = model(batch_x)
            loss = criterion(output, batch_y)
            batch_accuracy = calc_accuracy(output, batch_y)

            output = torch.sigmoid(output) >= 0.5

            y_true_1.extend(batch_y[:, 0].cpu().numpy())
            y_pred_1.extend(output[:, 0].cpu().numpy())
            
            y_true_2.extend(batch_y[:, 1].cpu().numpy())
            y_pred_2.extend(output[:, 1].cpu().numpy())

            batch_accuracy = calc_accuracy(output, batch_y)
            epoch_sum_train_accuracy += batch_accuracy

            epoch_sum_train_loss += loss.item()

    # save epoch stats for plotting
    epoch_train_loss = epoch_sum_train_loss / train_batches
    accuracy_test_log.append(epoch_sum_train_accuracy / train_batches)
    loss_test_log.append(epoch_train_loss)

    print(f"test results for {train_batches} batches, {total_test_matches} rows")

    # Calculate F1 scores
    f1_var1 = f1_score(y_true_1, y_pred_1)
    f1_var2 = f1_score(y_true_2, y_pred_2)

    
    

    cm1 = confusion_matrix(y_true_1, y_pred_1)
    cm2 = confusion_matrix(y_true_2, y_pred_2)

    _ = display_confusion_matrix(cm1, "output_1")
    _ = display_confusion_matrix(cm2, "output_2")

    print(f"\nDetailed Classification Report for: output_1")
    print(f1_var1)
    report1 = classification_report(y_true_1, y_pred_1, target_names=['Negative', 'Positive'])
    print(report1)
    print(f"\nDetailed Classification Report for: output_2")
    print(f1_var2)
    report2 = classification_report(y_true_2, y_pred_2, target_names=['Negative', 'Positive'])
    print(report2)

    
    # Calculate precision and recall
    precision_var1 = precision_score(y_true_1, y_pred_1)
    recall_var1 = recall_score(y_true_1, y_pred_1)
    
    precision_var2 = precision_score(y_true_2, y_pred_2)
    recall_var2 = recall_score(y_true_2, y_pred_2)
    
    # Package the metrics
    metrics_var1 = {
        'Precision': precision_var1,
        'Recall': recall_var1,
        'F1 Score': f1_var1
    }
    
    metrics_var2 = {
        'Precision': precision_var2,
        'Recall': recall_var2,
        'F1 Score': f1_var2
    }

    test_summary = {
        "test_rows": total_test_matches,
        "cm1": cm1,
        "cm2": cm2,
        "report1": str(report1),
        "report2": str(report2),
        "acc_log": accuracy_test_log,
        "metrics1": metrics_var1,
        "metrics2": metrics_var2,
    }

    with open(f"./training/{snake_case(config['model_name'])}_test_results.json", 'w') as f:
        json.dump(test_summary, f, indent=4)
    
    
# Create and display table for confusion matrix
def display_confusion_matrix(cm, output_name):
    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(
        cm, 
        index=['Actual Negative', 'Actual Positive'], 
        columns=['Predicted Negative', 'Predicted Positive']
    )
    
    print(f"\nConfusion Matrix for {output_name}:")
    print(cm_df)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics for {output_name}:")
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    print(metrics_df)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {output_name}')
    plt.tight_layout()
    plt.show()
    
    return metrics_df



if __name__ == "__main__":

    train()