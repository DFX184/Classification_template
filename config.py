from rich import print

parameter = {
    "ROOT": r"/mnt/d/dataset",
    "dataset_csv": r"train.csv",
    "batch_size": 16,
    "num_workers" : 8,
    "learning_rate" : 5e-5,
    "seed" : 12,
    "epochs":150,
    "num_classes" : 18,
    "in_channel" : 3,
    "image_size" : (256,256),
    "save_model" : "resnet18"
}

if __name__ == "__main__":
    print("paramters:",parameter)
