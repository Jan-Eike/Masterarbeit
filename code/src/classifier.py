import torch
import numpy as np
import lightgbm as lgb
from typing import Optional
from models import NNClassifier
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

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


    def __train_and_eval_nn(self, x_train, y_train, x_test, y_test, input_length, batch_size=10, lr=1e-4, seed=42, epochs=10) -> float:
        nn_classifier = NNClassifier(input_length)
        x_train = np.array(x_train, dtype="float32")
        dataset = TensorDataset(Tensor(x_train), Tensor(y_train))
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        torch.manual_seed(seed)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(nn_classifier.parameters(), lr=lr)
        for epoch in range(epochs):
            print(f'Starting epoch {epoch+1}')
            current_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                
                inputs, targets = data
                targets = targets.long()
                
                optimizer.zero_grad()
                
                outputs = nn_classifier(inputs)
                
                loss = loss_function(outputs, targets)
                
                loss.backward()
                
                optimizer.step()
                
                current_loss += loss.item()

        print('Training process has finished.')

        x_test = np.array(x_test, dtype="float32")
        dataset_test = TensorDataset(Tensor(x_test), Tensor(y_test))

        pred = []
        testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):

                inputs, targets = data
                targets = targets.long()

                outputs = nn_classifier(inputs)

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