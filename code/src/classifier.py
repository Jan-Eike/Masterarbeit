import torch
import numpy as np
import lightgbm as lgb
from typing import Optional
from models import NNClassifier, NNClassifierWithBERT
from pytorchtools import EarlyStopping
from pytorch_pretrained_bert import BertAdam
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from tqdm import tqdm

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

    def __create_datasets2(self, embedding_data, labels, tokenized_complete, batch_size):
        x_train, x_dev, x_test = embedding_data
        y_train, y_dev, y_test = labels
        text_data_train, text_data_dev, text_data_test = tokenized_complete
        x_train = np.array(x_train, dtype="float32")
        x_dev = np.array(x_dev, dtype="float32")
        x_test = np.array(x_test, dtype="float32")
        print(Tensor(text_data_dev["attention_mask"]).shape, Tensor(text_data_dev["input_ids"]).shape, Tensor(x_dev).shape, Tensor(y_dev).shape)
        train_data = TensorDataset(Tensor(text_data_train["input_ids"]).to("cuda"), Tensor(text_data_train["attention_mask"]).to("cuda"), Tensor(x_train).to("cuda"), Tensor(y_train).to("cuda"))
        validation_data = TensorDataset(Tensor(text_data_dev["input_ids"]).to("cuda"), Tensor(text_data_dev["attention_mask"]).to("cuda"), Tensor(x_dev).to("cuda"), Tensor(y_dev).to("cuda"))
        test_data = TensorDataset(Tensor(text_data_test["input_ids"]).to("cuda"), Tensor(text_data_test["attention_mask"]).to("cuda"), Tensor(x_test).to("cuda"), Tensor(y_test).to("cuda"))
        
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


    def train_and_eval_nn_complete(self, embedding_data, labels, tokenized_complete, input_length, batch_size=32, lr=2e-5, patience=10, seed=42, epochs=200) -> float:
        y_test = labels[2]
        model = NNClassifierWithBERT(input_length, 'bert-base-uncased').to("cuda")
        train_loader, valid_loader, test_loader = self.__create_datasets2(embedding_data, labels, tokenized_complete, batch_size)
        
        train_losses = []
        valid_losses = []
        avg_train_losses = []
        avg_valid_losses = [] 
        best_loss = np.inf
        torch.manual_seed(seed)
        loss_function = nn.CrossEntropyLoss()
        #loss_function = nn.BCEWithLogitsLoss()
        optimizer = BertAdam(model.parameters(), lr=lr, warmup=0.1, t_total=47 * epochs)
        #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in tqdm(range(epochs), leave=False):
            model.train()
            for batch, (token_ids, masks, embedding_inputs, targets) in enumerate(tqdm(train_loader, leave=False)):
                #targets = targets.float()
                targets = targets.long()
                token_ids = token_ids.long()
                optimizer.zero_grad()        
                outputs = model(token_ids, masks, embedding_inputs)
                #loss = loss_function(outputs, targets.reshape(targets.shape[0], 1))
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            model.eval()
            for token_ids, masks, embedding_inputs, targets in tqdm(valid_loader, leave=False):
                #targets = targets.float()
                targets = targets.long()
                token_ids = token_ids.long()
                outputs = model(token_ids, masks, embedding_inputs)
                #loss = loss_function(outputs, targets.reshape(targets.shape[0], 1))
                loss = loss_function(outputs, targets)
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), "best_model.pt")

            epoch_len = len(str(epochs))
            
            print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f}')

        
            print(print_msg)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

        print('Training process has finished.')

        # load the last checkpoint with the best model
        #model.load_state_dict(torch.load('checkpoint.pt'))
        model.load_state_dict(torch.load("best_model.pt"))
        pred = []
        with torch.no_grad():
            for batch, (token_ids, masks, embedding_inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
                token_ids = token_ids.long()
                outputs = model(token_ids, masks, embedding_inputs)

                _, predicted = torch.max(outputs, 1)
                pred.extend(predicted.tolist())

        f1_nn = f1_score(y_test, pred, average='macro')
        print(f1_nn)
        return f1_nn


    # adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    def __train_and_eval_nn(self, x_train, y_train, x_dev, y_dev, x_test, y_test, input_length, batch_size=32, lr=1e-4, patience=10, seed=42, epochs=200) -> float:
        model = NNClassifier(input_length)
        train_loader, valid_loader, test_loader = self.__create_datasets(x_train, y_train, x_dev, y_dev, x_test, y_test, batch_size)
        
        train_losses = []
        valid_losses = []
        avg_train_losses = []
        avg_valid_losses = [] 
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        torch.manual_seed(seed)
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

        f1_nn = f1_score(y_test, pred, average='macro')
        print(f1_nn)
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
        f1_lgbm = f1_score(y_test_df, (np.round(ypred)), average='macro')
        print(f1_lgbm)
        return f1_lgbm
        

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities, also not sure about just rounding...
    return 'f1', f1_score(y_true, y_hat, average='macro'), True