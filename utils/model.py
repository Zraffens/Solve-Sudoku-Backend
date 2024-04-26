import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # pooling layer
        self.pool = nn.MaxPool2d(2)
        # fully connected layers
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)     # Flatten
        x = (F.relu(self.fc1(x)))
        x = (F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class trainer():
    def __init__(self, model, train_dataloader, test_dataloader, device):
        self.device= device
        self.model= model.to(device)
        self.train_dataloader= train_dataloader
        self.test_dataloader= test_dataloader



    def train(self, epochs):
        '''trains the model using traditional approach'''
        criterion= torch.nn.CrossEntropyLoss().to(self.device)
        # criterion= torch.nn.BCEWithLogitsLoss().to(device)
        optimizer= torch.optim.Adam(self.model.parameters(), lr=0.001)
        running_loss_per_epoch=[]
        test_accuracies=[]
        minimum_val_loss= np.Inf #to track model weights with best validation performance
        for epoch in range(epochs):
            self.model.train()
            total_num=0
            running_loss= 0
            train_acc=0
            running_loss_per_batch= []
            start_time= time.perf_counter()
            for i, data in enumerate(self.train_dataloader):
                images, labels= data
                images= images.to(self.device)
                labels= labels.to(self.device)

                predicted_logits= self.model(images)
                # pred_labels= (F.sigmoid(predicted_logits)>0.5).int()
                # print(predicted_logits.shape)
                pred_labels= torch.argmax(predicted_logits, axis=1)
                # loss= criterion(predicted_logits, labels.type_as(predicted_logits))
                loss= criterion(predicted_logits, labels.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_acc += torch.sum(torch.eq(pred_labels, labels)).item()
                running_loss_per_batch.append(loss.item())
                total_num+= len(images)

            running_loss_per_epoch.append(np.mean(running_loss_per_batch))
            current_time= time.perf_counter()
            elapsed_time= current_time-start_time
            print("[epoch: %d] time elapsed: %.3f   loss: %.3f    train accuracy: %.3f" \
                    % (epoch + 1, elapsed_time, running_loss_per_epoch[-1], (train_acc/total_num)*100))
            test_stat= self.test()
            if test_stat['loss']< minimum_val_loss:
                minimum_val_loss=test_stat['loss']
                torch.save(self.model.state_dict(), 'checkpoint.pt' )
                print('Saved the latest best weights!!')
            test_accuracies.append(test_stat['accuracy'])

        return test_accuracies


    def test(self):
        # criterion= torch.nn.BCEWithLogitsLoss().to(device)
        criterion= torch.nn.CrossEntropyLoss().to(self.device)
        self.model.eval()  #setting the model in evaluation mode
        val_stat={}
        all_accuracy=[]
        all_loss=[]
        all_predictions=[]
        all_labels=[]
        total_num=0
        with torch.no_grad():
            running_loss_per_batch=[]
            for images, labels in self.test_dataloader:
                images, labels= images.to(self.device), labels.to(self.device)
                predicted_logits = self.model(images).float()
                labels_tensor = labels.clone().detach()
                # pred_labels= (F.sigmoid(pred_logits)>0.5).int()
                pred_labels= torch.argmax(predicted_logits, axis=1)
                # loss=criterion(pred_logits, labels_tensor.type_as(pred_logits))
                loss= criterion(predicted_logits, labels.long())
                accuracy= torch.sum(torch.eq(pred_labels, labels)).item()
                running_loss_per_batch.append(loss.item())    #tracking the loss, accuracy, predicted labels, and true labels
                all_accuracy.append(accuracy)
                all_predictions.append(pred_labels)
                all_labels.append(labels)
                total_num+=len(images)

        val_stat['loss'] = np.mean(running_loss_per_batch)
        val_stat['accuracy']=sum(all_accuracy)/total_num
        val_stat['prediction']=torch.cat(all_predictions, dim=0)
        val_stat['labels']=torch.cat(all_labels, dim=0)
        print(f"Test/Validation result: total sample: {total_num}, Avg loss: {val_stat['loss']:.3f}, Acc: {100*val_stat['accuracy']:.3f}%")
        return val_stat #returning the tracked values in the form of a dictionary



    