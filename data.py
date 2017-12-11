import csv
import os
from random import shuffle

from mxnet import gluon, nd

from mxnet import image


class IMDBWIKIDatasets(gluon.data.Dataset):
    def __init__(self, csv_path, train=True) -> None:
        self.train = train

        # csv content sample
        # /home/gdshen/datasets/face/imdb_crop/01/nm0000001_rm124825600_1899-5-10_1968.jpg,69
        # /home/gdshen/datasets/face/imdb_crop/01/nm0000001_rm3343756032_1899-5-10_1970.jpg,71
        # /home/gdshen/datasets/face/imdb_crop/01/nm0000001_rm577153792_1899-5-10_1968.jpg,69
        with open(csv_path, 'r') as csv_file:
            facereader = csv.reader(csv_file)
            self.image_age_list = [[row[0], int(row[1])] for row in
                                   list(facereader)]  # age read from csv is in type str, should convert to int

        shuffle(self.image_age_list)

    def __len__(self):
        return len(self.image_age_list)

    def __getitem__(self, index):
        img = image.imread(self.image_age_list[index][0], to_rgb=True)
        age = self.image_age_list[index][1]

        img = image.color_normalize(img.astype('float32') / 255, mean=nd.array([0.485, 0.456, 0.406]),
                                    std=nd.array([0.229, 0.224, 0.225]))
        img = image.imresize(img, 224, 224)
        img = nd.transpose(img, (2, 0, 1))
        age = nd.array([age]).asscalar().astype('float32')
        return img, age


class AsianFaceDatasets(gluon.data.Dataset):
    def __init__(self, csv_path, img_dir, train=True):
        self.train = train

        with open(csv_path, 'r') as csv_file:
            facereader = csv.reader(csv_file)
            facereader = list(facereader)[1:]
            self.img_age_list = [(os.path.join(img_dir, row[0].split('\\')[-1]), int(row[-1])) for row in facereader]
        if self.train:
            shuffle(self.img_age_list)

    def __len__(self):
        return len(self.img_age_list)

    def __getitem__(self, index):
        img = image.imread(self.img_age_list[index][0], to_rgb=True)
        age = self.img_age_list[index][1]

        img = image.color_normalize(img.astype('float32') / 255, mean=nd.array([0.485, 0.456, 0.406]),
                                    std=nd.array([0.229, 0.224, 0.225]))
        img = image.imresize(img, 224, 224)
        img = nd.transpose(img, (2, 0, 1))
        age = nd.array([age]).asscalar().astype('float32')
        return img, age
