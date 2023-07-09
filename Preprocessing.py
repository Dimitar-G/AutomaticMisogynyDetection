import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from Datasets import MAMIDataset, MAMIDatasetTesting
from Models import FusionModel
from Transforms import create_transform


DATASET_PATH = 'D:\\Datasets\\MAMI DATASET\\'
TRAINING_IMAGE_FOLDER = DATASET_PATH + 'training'
TRAINING_CSV_FILE = DATASET_PATH + 'training.csv'
VALIDATION_IMAGE_FOLDER = DATASET_PATH + 'trial'
VALIDATION_CSV_FILE = DATASET_PATH + 'trial.csv'
TESTING_IMAGE_FOLDER = DATASET_PATH + 'test'
TESTING_CSV_FILE = DATASET_PATH + 'test.csv'


def generate_embeddings_dict():
    # Training Data
    training_dataset = MAMIDataset(image_folder=TESTING_IMAGE_FOLDER, csv_file=TESTING_CSV_FILE,
                                   transform=create_transform())
    dataloader = DataLoader(training_dataset, batch_size=1, shuffle=False)
    print('Training dataset loaded.')

    # Validation Data
    # validation_dataset = MAMIDataset(image_folder=VALIDATION_IMAGE_FOLDER, csv_file=VALIDATION_CSV_FILE,
    #                                  transform=create_transform())
    # dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    # print('Validation dataset loaded.')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    embeddings = {'images': [], 'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}

    for i, (image, sentence, label) in tqdm(enumerate(dataloader)):
        embeddings['images'].append(image[0])
        sentence_embedding = tokenizer.encode_plus(sentence,
                                                   add_special_tokens=True,
                                                   truncation=True,
                                                   padding="max_length",
                                                   return_attention_mask=True,
                                                   return_tensors='pt')

        embeddings['input_ids'].append(sentence_embedding['input_ids'][0])
        embeddings['token_type_ids'].append(sentence_embedding['token_type_ids'][0])
        embeddings['attention_mask'].append(sentence_embedding['attention_mask'][0])
        embeddings['labels'].append(label)
        # print()

    torch.save(embeddings, 'embeddings/testing_preprocessed_dict.pt')


def generate_training_embeddings():
    data = torch.load('embeddings/testing_preprocessed_dict.pt')

    model = FusionModel()

    embeddings = []
    labels = []

    for image, input_id, token_type_id, attention_mask, label in tqdm(zip(data['images'], data['input_ids'], data['token_type_ids'], data['attention_mask'], data['labels']), total=len(data['images'])):

        # image embedding
        image_emb = model.efficient.features(torch.unsqueeze(image, 0))
        image_emb = model.efficient.avgpool(image_emb)
        image_emb = image_emb.view(image_emb.size(0), -1)

        # sentence embedding
        sentence_emb = model.bert(input_ids=torch.unsqueeze(input_id, 0), token_type_ids=torch.unsqueeze(token_type_id, 0), attention_mask=torch.unsqueeze(attention_mask, 0))
        sentence_emb = sentence_emb.pooler_output

        # fused embedding
        embedding = torch.cat((image_emb, sentence_emb), dim=1)

        embeddings.append(embedding)
        labels.append(label)

    embeddings_dict = {'embeddings': embeddings, 'labels': labels}
    torch.save(embeddings_dict, 'embeddings/testing_embeddings.pt')


def generate_embeddings_dict_testing():
    # Testing Data
    testing_dataset = MAMIDatasetTesting(image_folder=TESTING_IMAGE_FOLDER, csv_file=TESTING_CSV_FILE)
    dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)
    print('Testing dataset loaded.')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    embeddings = {'images': [], 'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}

    for i, (image, sentence) in tqdm(enumerate(dataloader)):
        embeddings['images'].append(image[0])
        sentence_embedding = tokenizer.encode_plus(sentence,
                                                   add_special_tokens=True,
                                                   truncation=True,
                                                   padding="max_length",
                                                   return_attention_mask=True,
                                                   return_tensors='pt')

        embeddings['input_ids'].append(sentence_embedding['input_ids'][0])
        embeddings['token_type_ids'].append(sentence_embedding['token_type_ids'][0])
        embeddings['attention_mask'].append(sentence_embedding['attention_mask'][0])
        # print()

    torch.save(embeddings, 'embeddings/testing_preprocessed_dict.pt')


def generate_testing_embeddings():
    data = torch.load('embeddings/testing_preprocessed_dict.pt')

    model = FusionModel()

    embeddings = []

    for image, input_id, token_type_id, attention_mask in tqdm(zip(data['images'], data['input_ids'], data['token_type_ids'], data['attention_mask']), total=len(data['images'])):

        # image embedding
        image_emb = model.efficient.features(torch.unsqueeze(image, 0))
        image_emb = model.efficient.avgpool(image_emb)
        image_emb = image_emb.view(image_emb.size(0), -1)

        # sentence embedding
        sentence_emb = model.bert(input_ids=torch.unsqueeze(input_id, 0), token_type_ids=torch.unsqueeze(token_type_id, 0), attention_mask=torch.unsqueeze(attention_mask, 0))
        sentence_emb = sentence_emb.pooler_output

        # fused embedding
        embedding = torch.cat((image_emb, sentence_emb), dim=1)

        embeddings.append(embedding)

    embeddings_dict = {'embeddings': embeddings}
    torch.save(embeddings_dict, 'embeddings/testing_embeddings.pt')


if __name__ == '__main__':
    generate_testing_embeddings()
