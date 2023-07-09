import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from Datasets import EmbeddingDataset
from Models import FCModel2
from statistics import mean


# adjust before running script
TRAINING_EMBEDDINGS_PATH = './embeddings/training_embeddings.pt'
VALIDATION_EMBEDDINGS_PATH = './embeddings/validation_embeddings.pt'

MODELS_FOLDER = './models8'
INFO_FILE = MODELS_FOLDER + '/info.csv'

LOAD_MODEL = './models4/model_977.pt'  # None if no weights should be loaded


if __name__ == '__main__':

    if not os.path.exists(MODELS_FOLDER):
        os.mkdir(MODELS_FOLDER)

    if not os.path.exists(INFO_FILE):
        open(INFO_FILE, 'w').close()

    # Training Data
    training_dataset = EmbeddingDataset(TRAINING_EMBEDDINGS_PATH)
    training_dataloader = DataLoader(training_dataset, batch_size=128, shuffle=True, num_workers=2)
    print('Training dataset loaded.')

    # Validation Data
    validation_dataset = EmbeddingDataset(VALIDATION_EMBEDDINGS_PATH)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=2)
    print('Validation dataset loaded.')

    model = FCModel2()
    if LOAD_MODEL is not None:
        model.load_state_dict(torch.load(LOAD_MODEL))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    num_epochs = 500
    max_val_acc = 0

    for epoch in range(num_epochs):

        loss_epoch = []
        for i, (embeddings, labels) in tqdm(enumerate(training_dataloader), desc=f'Batches in epoch {epoch}'):
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels.float())
            loss_epoch.append(float(loss.clone().detach().numpy()))
            loss.backward()
            optimizer.step()

        training_loss = mean(loss_epoch)

        # Evaluate the model on the validation set
        # print('Validating:')
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for i, (embeddings, labels) in enumerate(validation_dataloader):
                output = model(embeddings)
                val_loss += criterion(output, labels.float()).item()
                pred = (output > torch.tensor([0.5])).float() * 1
                val_correct += pred.eq(labels.view_as(pred)).sum().item()

        # Print the results for the current epoch
        val_loss /= len(validation_dataset)
        val_acc = 100. * val_correct / len(validation_dataset)
        print(f'Validation accuracy: {val_acc}')
        print(f'Validation loss: {val_loss}')

        with open(INFO_FILE, 'a') as file:
            file.write(f'{epoch},{training_loss},{val_loss},{val_acc}\n')
            file.flush()

        if val_acc > max_val_acc or epoch % 5 == 0:
            max_val_acc = val_acc
            torch.save(model.state_dict(), f"{MODELS_FOLDER}/model_{epoch}.pt")

    torch.save(model.state_dict(), f"{MODELS_FOLDER}/model_{num_epochs}.pt")
