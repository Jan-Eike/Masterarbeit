import torch
import numpy as np
import lightgbm as lgb
import transformers
import datasets
from models import NNClassifier, NNClassifierWithBert
from transformers import IntervalStrategy, set_seed
from pytorchtools import EarlyStopping
from torch import nn, Tensor
from torch.utils.data import TensorDataset
from sklearn.metrics import f1_score
from utils import (
    preprocess_logits_for_metrics,
    compute_metrics,
    lgb_f1_score
)

class Classifier():
    def __init__(self, class_type: str) -> None:
        self.class_type = class_type


    def train_and_eval(self, params, *data) -> float:
        f1 = -1
        if self.class_type == "nn":
            f1 = self.__train_and_eval_nn(*data, **params)
        elif self.class_type == "lgbm":
            f1 = self.__train_and_eval_lgbm(*data, params)
        else:
            print(f"Classification type {self.class_type} not found!")
        return f1


    def __create_datasets(self, x_train, y_train, x_dev, y_dev, x_test, y_test, batch_size):
        x_train = np.array(x_train, dtype="float32")
        x_dev = np.array(x_dev, dtype="float32")
        x_test = np.array(x_test, dtype="float32")
        train_data = TensorDataset(Tensor(x_train), Tensor(y_train))
        validation_data = TensorDataset(Tensor(x_dev), Tensor(y_dev))
        test_data = TensorDataset(Tensor(x_test), Tensor(y_test))
        
        # load training data in batches
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                num_workers=0)
        
        # load validation data in batches
        valid_loader = torch.utils.data.DataLoader(validation_data,
                                                batch_size=batch_size,
                                                num_workers=0)
        
        # load test data in batches
        test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size,
                                                num_workers=0)
        return train_loader, valid_loader, test_loader


    def train_and_eval_nn_complete(self, embedding_data, save, tokenized_complete, seed, input_length, model_attributes) -> float:
        tokenized_train, tokenized_dev, tokenized_test = tokenized_complete
        x_train, x_dev, x_test = embedding_data
        dataset_custom_embeddings_train = datasets.Dataset.from_dict({"custom_embeddings": x_train})
        dataset_custom_embeddings_dev = datasets.Dataset.from_dict({"custom_embeddings": x_dev})
        dataset_custom_embeddings_test = datasets.Dataset.from_dict({"custom_embeddings": x_test})
        tokenized_train = datasets.concatenate_datasets([tokenized_train, dataset_custom_embeddings_train], axis=1)
        tokenized_dev = datasets.concatenate_datasets([tokenized_dev, dataset_custom_embeddings_dev], axis=1)
        tokenized_test = datasets.concatenate_datasets([tokenized_test, dataset_custom_embeddings_test], axis=1)
        # save instead of model_attributes.model_checkpoint
        set_seed(seed)
        model_class = NNClassifierWithBert(model_attributes.model_checkpoint, 2, input_length).to("cuda")
        training_args = transformers.TrainingArguments(
            output_dir=save,
            overwrite_output_dir=False,
            per_device_train_batch_size=4, # change if not enough cuda memory
            per_device_eval_batch_size=4, # change if not enough cuda memory
            gradient_accumulation_steps=16,
            num_train_epochs=3,
            warmup_steps=100,
            load_best_model_at_end=True,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=250, # 250
            save_total_limit=10,
            save_steps=250, # 250
        )
        trainer = transformers.Trainer(
            model=model_class,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            args=training_args
        )
        trainer.train()
        trainer.model.eval()
        baseline_scores = trainer.evaluate(eval_dataset=tokenized_test)
        trainer.save_model(output_dir=save+"/LM_classification_complete")
        print("f1 complete: {}".format(baseline_scores["eval_f1"]))
        return baseline_scores["eval_f1"]


    # adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    def __train_and_eval_nn(self, x_train, y_train, x_dev, y_dev, x_test, y_test, input_length, batch_size=32, lr=1e-4, patience=10, seed=42, epochs=200) -> float:
        set_seed(seed)
        model = NNClassifier(input_length)
        train_loader, valid_loader, test_loader = self.__create_datasets(x_train, y_train, x_dev, y_dev, x_test, y_test, batch_size)
        
        train_losses = []
        valid_losses = []
        avg_train_losses = []
        avg_valid_losses = [] 
        early_stopping = EarlyStopping(patience=patience, verbose=False)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            for batch, (inputs, targets) in enumerate(train_loader, 0):
                targets = targets.long()
                optimizer.zero_grad()        
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            model.eval()
            for data, targets in valid_loader:
                targets = targets.long()
                outputs = model(data)
                loss = loss_function(outputs, targets)
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            epoch_len = len(str(epochs))
            
            print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f}')

            if epoch % 10 == 0:
                print(print_msg)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('Training process has finished.')

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))

        pred = []
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):

                inputs, targets = data
                targets = targets.long()

                outputs = model(inputs)

                _, predicted = torch.max(outputs, 1)
                pred.extend(predicted.tolist())

        f1_nn = f1_score(y_test, pred)
        print(f"f1: {f1_nn}")
        return f1_nn


    def __train_and_eval_lgbm(self, x_train_df, y_train_df, x_test_df, y_test_df, x_dev_df, y_dev_df, num_round, params):
        lgb_model = lgb.train(
            params,
            lgb.Dataset(x_train_df, label=y_train_df),
            num_round, 
            valid_sets=[lgb.Dataset(x_dev_df, label=y_dev_df)], 
            callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(100)], 
            feval=lgb_f1_score
        )
        
        ypred = lgb_model.predict(x_test_df, num_iteration=lgb_model.best_iteration)
        f1_lgbm = f1_score(y_test_df, (np.round(ypred)))
        print(f"f1: {f1_lgbm}")
        return f1_lgbm