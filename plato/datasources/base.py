"""
Base class for data sources, encapsulating training and testing datasets with
custom augmentations and transforms already accommodated.
"""
import gzip
import logging
import os
import sys
import tarfile
import zipfile
from urllib.parse import urlparse

import requests
from plato.config import Config


class DataSource:
    """
    Training and testing datasets with custom augmentations and transforms
    already accommodated.
    """
    def __init__(self):
        self.trainset = None
        self.testset = None

    @staticmethod
    def download(url, data_path):
        """downloads a dataset from a URL."""
        if not os.path.exists(data_path):
            if Config().clients.total_clients > 1:
                if not hasattr(Config().data, 'concurrent_download'
                               ) or not Config().data.concurrent_download:
                    raise ValueError(
                        "The dataset has not yet been downloaded from the Internet. "
                        "Please re-run with '-d' or '--download' first. ")

            os.makedirs(data_path, exist_ok=True)

        url_parse = urlparse(url)
        file_name = os.path.join(data_path, url_parse.path.split('/')[-1])

        if not os.path.exists(file_name.replace('.gz', '')):
            logging.info("Downloading %s.", url)

            res = requests.get(url, stream=True)
            total_size = int(res.headers["Content-Length"])
            downloaded_size = 0

            with open(file_name, "wb+") as file:
                for chunk in res.iter_content(chunk_size=1024):
                    downloaded_size += len(chunk)
                    file.write(chunk)
                    file.flush()
                    sys.stdout.write("\r{:.1f}%".format(100 * downloaded_size /
                                                        total_size))
                    sys.stdout.flush()
                sys.stdout.write("\n")

            # Unzip the compressed file just downloaded
            logging.info("Decompressing the dataset downloaded.")
            name, suffix = os.path.splitext(file_name)

            if file_name.endswith("tar.gz"):
                tar = tarfile.open(file_name, "r:gz")
                tar.extractall(data_path)
                tar.close()
                os.remove(file_name)
            elif suffix == '.zip':
                logging.info("Extracting %s to %s.", file_name, data_path)
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall(data_path)
            elif suffix == '.gz':
                unzipped_file = open(name, 'wb')
                zipped_file = gzip.GzipFile(file_name)
                unzipped_file.write(zipped_file.read())
                zipped_file.close()
                os.remove(file_name)
            else:
                logging.info("Unknown compressed file type.")
                sys.exit()

        if Config().args.download:
            logging.info("The dataset has been successfully downloaded. "
                         "Re-run the experiment without '-d' or '--download'.")
            sys.exit()

    def num_train_examples(self) -> int:
        """ Obtains the number of training examples. """
        return len(self.trainset)

    def num_test_examples(self) -> int:
        """ Obtains the number of testing examples. """
        return len(self.testset)

    def classes(self):
        """ Obtains a list of class names in the dataset. """
        ###################################################################
        if(self.trainset is None):
            self.trainset = self.get_train_set()
        from plato.datasources.yolov5.utils.dataloaders import LoadImagesAndLabels
        if(isinstance(self.trainset, LoadImagesAndLabels)):
            # return ["0", "1"]
            return ["fall", "not_fall"] ### 这里也需要改成troch.Tensor么? 但张量里好像不能装字符串
            # return [0, 1] ######### ???
        ###################################################################
        return list(self.trainset.classes)

    def targets(self):
        """ Obtains a list of targets (labels) for all the examples
        in the dataset. """
        ###################################################################
        if(self.trainset is None):
            self.trainset = self.get_train_set()
        print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz %s", type(self.trainset))
        from plato.datasources.yolov5.utils.dataloaders import LoadImagesAndLabels
        import torch
        if(isinstance(self.trainset, LoadImagesAndLabels)):
            # return ["fall", "not_fall"]
            # return ["0", "1"]
            # return [0, 1] ######### ???
            # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            # return torch.IntTensor([0, 1]) ######### ???
            # tmp = torch.IntTensor([
            #     0,
            #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #     0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
            #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            # ])
            # print(type(tmp))
            # print(tmp)
            # return torch.IntTensor([
            #     0,
            #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #     0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
            #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            # ])

            # print(type(self.trainset.labels))   
            # print(self.trainset.labels) # 有类别有位置 [array([[0, 0.45156, 0.675, 0.37813, 0.35833]], dtype=float32), array([[0, 0.45312, 0.69583, 0.38125, 0.35]], dtype=float32) ....
            # return self.trainset.labels

            # print(type(self.trainset.labels[:, 0]))
            # print(self.trainset.labels[:, 0])
            # return self.trainset.labels[:, 0]

            # print("@@@@@@")        
            ret = []
            for i in self.trainset.labels:
                ret.append(int(i[0][0]))
            ret = torch.IntTensor(ret)
            print(type(ret))
            print(ret)
            return ret
        ###################################################################
        return self.trainset.targets

    def get_train_set(self):
        """ Obtains the training dataset. """
        return self.trainset

    def get_test_set(self):
        """ Obtains the validation dataset. """
        return self.testset
