import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import BertModel, BertTokenizer
from Datasets import MAMIDataset
from tqdm import tqdm
from Datasets import MAMIDataset
from Models import FusionModel
from statistics import mean


DATASET_PATH = 'D:\\Datasets\\MAMI DATASET\\'
TRAINING_IMAGE_FOLDER = DATASET_PATH + 'training'
TRAINING_CSV_FILE = DATASET_PATH + 'training.csv'
VALIDATION_IMAGE_FOLDER = DATASET_PATH + 'trial'
VALIDATION_CSV_FILE = DATASET_PATH + 'trial.csv'


if __name__ == '__main__':

    # Training Data
    training_dataset = MAMIDataset(image_folder=TRAINING_IMAGE_FOLDER, csv_file=TRAINING_CSV_FILE)
    training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=2)
    print('Training dataset loaded.')

    # Validation Data
    validation_dataset = MAMIDataset(image_folder=VALIDATION_IMAGE_FOLDER, csv_file=VALIDATION_CSV_FILE)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=2)
    print('Validation dataset loaded.')

    model = FusionModel()
    # model.load_state_dict(torch.load('efficient_clf_500.pt'))
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.00001)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    num_epochs = 200

    for epoch in range(num_epochs):

        loss_epoch = []
        for i, (images, sentences, labels) in tqdm(enumerate(training_dataloader), desc=f'Batches in epoch {epoch}'):
            optimizer.zero_grad()
            sentences = tokenizer.batch_encode_plus(sentences,
                                                    add_special_tokens=True,
                                                    truncation=True,
                                                    padding="max_length",
                                                    return_attention_mask=True,
                                                    return_tensors='pt')
            outputs = model(images, sentences)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss_epoch.append(loss.numpy()[0])
            loss.backward()
            optimizer.step()

        training_loss = mean(loss_epoch)

        # Evaluate the model on the validation set
        # print('Validating:')
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for batch_idx, (images, sentences, labels) in enumerate(validation_dataloader):
                sentences = tokenizer.batch_encode_plus(sentences,
                                                        add_special_tokens=True,
                                                        truncation=True,
                                                        padding="max_length",
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
                output = model(images, sentences)
                val_loss += criterion(output, labels.float().unsqueeze(1)).item()
                pred = (output > torch.tensor([0.5])).float() * 1
                val_correct += pred.eq(labels.view_as(pred)).sum().item()

        # Print the results for the current epoch
        val_loss /= len(validation_dataset)
        val_acc = 100. * val_correct / len(validation_dataset)
        print(f'Validation accuracy: {val_acc}')
        print(f'Validation loss: {val_loss}')

        with open('training_info.csv', 'a') as file:
            file.write(f'{epoch},{training_loss},{val_loss},{val_acc}\n')
            file.flush()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./models/model_{epoch}.pt")

    torch.save(model.state_dict(), f"model_{num_epochs}.pt")
