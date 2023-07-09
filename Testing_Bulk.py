import torch
from torch.utils.data import DataLoader
import os
from Datasets import EmbeddingDataset, EmbeddingDatasetTesting
from Models import FCModel, FCModel2
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm


# adjust before running script
TESTING_EMBEDDINGS_PATH = './embeddings/testing_embeddings.pt'
VALIDATION_EMBEDDINGS_PATH = './embeddings/validation_embeddings.pt'

LOAD_MODELS_FOLDER = './models'

OUTPUT_FOLDER = './test_results'
INFO_FILE = OUTPUT_FOLDER + LOAD_MODELS_FOLDER[1:] + '.csv'


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    if not os.path.exists(INFO_FILE):
        open(INFO_FILE, 'w').close()

    # Testing Data
    testing_dataset = EmbeddingDatasetTesting(TESTING_EMBEDDINGS_PATH)
    print('Testing dataset loaded.')

    for m in tqdm(list(filter(lambda f: f.startswith('model'), os.listdir(LOAD_MODELS_FOLDER)))):

        model = FCModel()
        model.load_state_dict(torch.load(f'{LOAD_MODELS_FOLDER}/{m}'))

        testing_dataloader = DataLoader(testing_dataset, batch_size=1000, num_workers=2)

        with torch.no_grad():

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
            file.write(f'{m}: \t precision={test_precision} \t recall={test_recall} \t accuracy={test_accuracy} \t '
                       f'f1_score={test_f1} \n')
            file.flush()
