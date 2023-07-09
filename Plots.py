import pandas as pd
import matplotlib.pyplot as plt

FOLDER_PATH = './models8'


def generate_plots(folder_path):
    df = pd.read_csv(f'{folder_path}/info.csv', header=None)

    # training loss
    plt.plot(df[0], df[1])
    plt.title("Training loss")
    plt.savefig(f'{folder_path}/train_loss.png')
    plt.clf()

    # validation loss
    plt.plot(df[0], df[2])
    plt.title("Validation loss")
    plt.savefig(f'{folder_path}/val_loss.png')
    plt.clf()

    # validation accuracy
    plt.plot(df[0], df[3])
    plt.title("Validation accuracy")
    plt.savefig(f'{folder_path}/val_acc.png')
    plt.clf()


if __name__ == '__main__':
    generate_plots(FOLDER_PATH)
