import torch
from torch.utils.data import DataLoader
import os
from Datasets import EmbeddingDataset, EmbeddingDatasetTesting
from Models import FCModel, FCModel2
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


# adjust before running script
TESTING_EMBEDDINGS_PATH = './embeddings/testing_embeddings.pt'
VALIDATION_EMBEDDINGS_PATH = './embeddings/validation_embeddings.pt'

LOAD_MODEL = './models7/model_221.pt'

OUTPUT_FOLDER = './test_results'
INFO_FILE = OUTPUT_FOLDER + f'/{"_".join(LOAD_MODEL.split("/")[1:])}.csv'


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    if not os.path.exists(INFO_FILE):
        open(INFO_FILE, 'w').close()

    # Training Data
    testing_dataset = EmbeddingDatasetTesting(TESTING_EMBEDDINGS_PATH)
    testing_dataloader = DataLoader(testing_dataset, batch_size=1000, shuffle=True, num_workers=2)
    print('Testing dataset loaded.')

    # Validation Data
    validation_dataset = EmbeddingDataset(VALIDATION_EMBEDDINGS_PATH)
    validation_dataloader = DataLoader(validation_dataset, batch_size=100, shuffle=True, num_workers=2)
    print('Validation dataset loaded.')

    model = FCModel()
    model.load_state_dict(torch.load(LOAD_MODEL))

    with torch.no_grad():
        # Evaluate the model on the validation set
        embeddings, labels = next(iter(validation_dataloader))
        output = model(embeddings)
        pred = (output > torch.tensor([0.5])).float() * 1

        val_precision = precision_score(labels, pred)
        val_recall = recall_score(labels, pred)
        val_accuracy = accuracy_score(labels, pred)
        val_f1 = f1_score(labels, pred)

        # Evaluate the model on the testing set
        embeddings, labels = next(iter(testing_dataloader))
        output = model(embeddings)
        pred = (output > torch.tensor([0.5])).float() * 1

        test_precision = precision_score(labels, pred)
        test_recall = recall_score(labels, pred)
        test_accuracy = accuracy_score(labels, pred)
        test_f1 = f1_score(labels, pred)

    # Save results
    with open(INFO_FILE, 'a') as file:
        file.write(f'Validation: \t precision={val_precision} \t recall={val_recall} \t accuracy={val_accuracy} \t '
                   f'f1_score={val_f1} \n')
        file.write(f'Testing: \t precision={test_precision} \t recall={test_recall} \t accuracy={test_accuracy} \t '
                   f'f1_score={test_f1} \n')
        file.flush()
