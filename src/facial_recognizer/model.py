from __future__ import annotations

import torch
import torch.nn.functional as F
import torchinfo
import torchvision
from torch import nn

# example model https://www.kaggle.com/code/strangetravel/facial-recog-siamese-network#Introduction


def compute_accuracy(distance, threshhold, true_labels):
    # all predictions that are below 0.75 are a match
    y = true_labels.float()
    prediction = distance.to(torch.device('cpu')).detach().apply_(lambda x: 0 if x < threshhold else 1)
    return sum(prediction == y.to(torch.device('cpu'))) / len(true_labels)


class SiameseNetwork(nn.Module):
    def __init__(self, train_backbone_params=True):
        super().__init__()
        self._train_backbone_params = train_backbone_params

        self.model = None
        # self._backbone = None

        self.loss = ContrastiveLoss()
        self._init_backbone()
        self._build_siamese()

    def _init_backbone(self):
        # Load a pre-trained ResNet-18 model from the torchvision library
        backbone = torchvision.models.resnet18(pretrained=True)

        # Freeze or unfreeze the parameters of the backbone depending on the train_backbone_params argument
        if self._train_backbone_params:
            for param in backbone.parameters():
                param.requires_grad = True
        else:
            for param in backbone.parameters():
                param.requires_grad = False

        self._backbone = backbone

    def _build_siamese(self):
        # Replace the last fully-connected layer of the ResNet-18 model with a new one that outputs a feature vector of size 128
        self.model = self._backbone
        self.model.fc = nn.Sequential(
            nn.Linear(self._backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # final feature vector
        )

    def forward_once(self, X):
        output = self.model(X)
        return output

    def forward(self, X1, X2):
        # Pass each input image through the model to obtain its feature vector
        output1 = self.forward_once(X1)
        output2 = self.forward_once(X2)
        distance = F.pairwise_distance(output1, output2)
        return distance

    def training_step(self, batch, threshhold=0.75):
        X, y = batch
        out = self(X[0], X[1])
        loss = self.loss(out, y)
        acc = compute_accuracy(out, threshhold, y)

        return loss, acc

    def validation_step(self, batch, threshhold=0.75):
        X, y = batch
        out = self(X[0], X[1])
        val_loss = self.loss(out, y)
        val_acc = compute_accuracy(out, threshhold, y)
        return {'val_loss': val_loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        batch_acc = [x['val_acc'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean().item()
        epoch_acc = torch.stack(batch_acc).mean().item()
        return {'val_loss': epoch_loss, 'val_acc': epoch_acc}

    def evaluate(self, dl):
        self.eval()
        with torch.no_grad():
            self.eval()
            outputs = [self.validation_step(batch) for batch in dl]

        return self.validation_epoch_end(outputs)

    def epoch_end_val(self, epoch, results):
        print(f"Epoch:[{epoch}]: validation loss: {results['val_loss']}, validation accuracy: {results['val_acc']}")


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin

    def forward(self, distance, labels):
        # Calculate the Euclidean distance between the two feature vectors
        y = labels.float()

        loss = torch.mean(
            torch.square(distance) * (1 - y)
            + torch.square(torch.max(self.margin - distance, torch.zeros_like(distance))) * (y),
        )
        return loss

    # training function


def fit(model, epochs, batch_size, train_generator, val_generator, optimizer, learning_rate, lr_scheduler, **kwargs):
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # set up learning rate scheduler if desired
    if lr_scheduler:
        lrs = lr_scheduler(optimizer, **kwargs)

    # set up list to log data of training
    history = []
    min_val_loss = float('inf')

    # set model to train mode to activate layers specific for training, such as the dropout layer
    model.train()
    for epoch in range(epochs):
        # set up list of metrics for epoch which are updated after each batch
        train_losses = []
        train_acc = []
        for num, batch in enumerate(train_generator(batch_size)):
            lossF = ContrastiveLoss(margin=3)
            optimizer.zero_grad()
            print(f'New batch [{num}]')
            loss, acc = model.training_step(batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach())
            train_acc.append(acc)
            print(f'Batch loss: {loss}')
            print(f'Batch accuracy: {acc}')
            # break for loop if num_batches is reached
            # if num > (int(np.ceil(len(train_positives) / batch_size)) - 1):
            #    break

        result = model.evaluate(val_generator(batch_size))
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = torch.stack(train_acc).mean().item()

        history.append(result)

        if lr_scheduler:
            lrs.step(metrics=result['val_loss'])
            print('Decreased!')

        model.epoch_end_val(epoch, result)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"Train loss epoch: {result['train_loss']}")
        print(f"Train accuracy epoch: {result['train_acc']}")

        # save best model
        if result['val_loss'] < min_val_loss:
            torch.save(model, 'best_model.pt')
            min_val_loss = result['val_loss']

    return history


if __name__ == '__main__':
    # instantiate model
    # get summary

    device = 'cuda'
    model = SiameseNetwork()
    model = model.to(device)
    print(torchinfo.summary(model))

    # fit the model using the predefined fit function
    fit_it = True
    if fit_it:
        torch.cuda.empty_cache()  # empty lefover cuda cache
        history = fit(
            model,
            100,
            128,
            train_generator,
            test_generator,
            torch.optim.Adam,
            0.001,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            patience=5,
            factor=0.05,
        )
        torch.cuda.empty_cache()  # empty cuda cache after training to prevent memory error
