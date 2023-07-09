import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from Transforms import create_transform_augment, create_transform
from transformers import BertTokenizer, BertModel


class MAMIDataset(Dataset):
    """
    This class is a custom defined Dataset for this project.
    """

    def __init__(self, image_folder, csv_file, transform=None, tokenizer=None):
        if transform is None:
            self.transform = create_transform_augment(256)
        else:
            self.transform = transform
        # if tokenizer is None:
        #     self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # else:
        #     self.tokenizer = tokenizer

        df = pd.read_csv(csv_file, sep='\t')
        df['file_path'] = df['file_name'].apply(lambda name: os.path.join(image_folder, name))
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.loc[index]
        image = Image.open(row['file_path'])
        misogynous = row['misogynous']
        transcription = row['Text Transcription']

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        # transcription = self.tokenizer.encode_plus(transcription,
        #                                            add_special_tokens=True,
        #                                            truncation=True,
        #                                            padding="max_length",
        #                                            return_attention_mask=True,
        #                                            return_tensors='pt')

        return image, transcription, misogynous


class MAMIDatasetTesting(Dataset):
    """
    This class is a custom defined Dataset for this project.
    """

    def __init__(self, image_folder, csv_file, transform=None, tokenizer=None):
        if transform is None:
            self.transform = create_transform(256)
        else:
            self.transform = transform

        df = pd.read_csv(csv_file, sep='\t')
        df['file_path'] = df['file_name'].apply(lambda name: os.path.join(image_folder, name))
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.loc[index]
        image = Image.open(row['file_path'])
        transcription = row['Text Transcription']

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)

        return image, transcription


class EmbeddingDataset(Dataset):

    def __init__(self, file_path):
        self.data = torch.load(file_path)

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, index):
        return self.data['embeddings'][index][0], self.data['labels'][index]


class EmbeddingDatasetTesting(Dataset):

    def __init__(self, file_path):
        self.data = torch.load(file_path)
        labels_raw = pd.read_csv('D:\\Datasets\\MAMI DATASET\\test_labels.txt', sep='\t').iloc[:, 1]
        self.labels = torch.tensor([[float(label)] for label in labels_raw])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data['embeddings'][index][0], self.labels[index]


if __name__ == '__main__':
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #
    # dataset_path = 'D:\\Datasets\\MAMI DATASET\\'
    # image_folder = dataset_path + 'training'
    # csv_file = dataset_path + 'training.csv'
    # dataset = MAMIDataset(image_folder=image_folder, csv_file=csv_file, transform=create_transform_augment(224))
    #
    # sample = dataset[0]
    #
    # bert = BertModel.from_pretrained('bert-base-uncased')
    # binput = sample[1]
    # boutput = bert(**binput)

    # dataset = EmbeddingDataset('./embeddings/validation_embeddings.pt')

    ...
