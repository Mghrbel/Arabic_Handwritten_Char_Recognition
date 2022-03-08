from data_downloader import download_dataset, create_labeled_trained_dirs
from model import get_OCR_model
from model_trainer import train

if __name__ == '__main__' :
    download_dataset()
    create_labeled_trained_dirs()
    model = get_OCR_model()
    train(model)