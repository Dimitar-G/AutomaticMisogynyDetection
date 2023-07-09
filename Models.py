import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import BertModel, BertTokenizer
from Datasets import MAMIDataset


class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        # BERT module for text data (output size: self.bert.config.hidden_size)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # EfficientNet module for image data
        self.efficient = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # fully connected trainable layers
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 516)
        self.fc3 = nn.Linear(516, 258)
        self.fc4 = nn.Linear(258, 1)

        # freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        # freeze EfficientNet parameters
        for param in self.efficient.parameters():
            param.requires_grad = False

    def forward(self, transformed_image, tokenized_sentence):
        x1 = self.efficient.features(transformed_image)
        x1 = self.efficient.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        # image embedding size: 1280

        x2 = self.bert(**tokenized_sentence)
        x2 = x2.pooler_output
        # text embedding size: 768

        # concatenating image vector and text vector
        x = torch.cat((x1, x2), dim=1)
        # fused embedding size: 2048

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x


class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()

        # fully connected trainable layers
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 516)
        self.fc3 = nn.Linear(516, 258)
        self.fc4 = nn.Linear(258, 1)

    def forward(self, embedding):
        # fused embedding size: 2048

        x = F.relu(self.fc1(embedding))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x


class FCModel2(nn.Module):
    def __init__(self):
        super(FCModel2, self).__init__()

        # fully connected trainable layers
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 1)

    def forward(self, embedding):
        # fused embedding size: 2048

        x = F.relu(self.fc1(embedding))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))

        return x


if __name__ == '__main__':
    model = FusionModel()

    dataset_path = 'D:\\Datasets\\MAMI DATASET\\'
    image_folder = dataset_path + 'training'
    csv_file = dataset_path + 'training.csv'
    dataset = MAMIDataset(image_folder=image_folder, csv_file=csv_file)
    dataloader = DataLoader(dataset, batch_size=3)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for batch_idx, (images, sentences, labels) in enumerate(dataloader):
        sentences = tokenizer.batch_encode_plus(sentences,
                                                add_special_tokens=True,
                                                truncation=True,
                                                padding="max_length",
                                                return_attention_mask=True,
                                                return_tensors='pt')
        embeddings = model(images, sentences)
        break
    ...