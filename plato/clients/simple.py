"""
A basic federated learning client who sends weight updates to the server.
"""

import logging
import time
from dataclasses import dataclass

from plato.algorithms import registry as algorithms_registry
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.samplers import registry as samplers_registry
from plato.trainers import registry as trainers_registry

from plato.clients import base as base


@dataclass
class Report(base.Report):
    """Report from a simple client, to be sent to the federated learning server."""
    training_time: float
    data_loading_time: float


@dataclass
class OortReport(Report):
    moving_loss_norm: float


class Client(base.Client):
    """A basic federated learning client who sends simple weight updates."""
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__()
        self.model = model
        self.datasource = datasource
        self.algorithm = algorithm
        self.trainer = trainer
        self.trainset = None  # Training dataset
        self.testset = None  # Testing dataset
        self.sampler = None

        self.data_loading_time = None
        self.data_loading_time_sent = False

    def __repr__(self):
        return 'Client #{}.'.format(self.client_id)

    def configure(self) -> None:
        """Prepare this client for training."""
        logging.info("*********************** simple.py Client.configure *********************")
        if self.trainer is None:
            self.trainer = trainers_registry.get(self.model)
        self.trainer.set_client_id(self.client_id)

        if self.algorithm is None:
            self.algorithm = algorithms_registry.get(self.trainer)
        self.algorithm.set_client_id(self.client_id)

    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        data_loading_start_time = time.perf_counter()
        logging.info("[Client #%d] Loading its data source...", self.client_id)

        if self.datasource is None:
            self.datasource = datasources_registry.get(
                client_id=self.client_id)

        self.data_loaded = True

        logging.info("[Client #%d] Dataset size: %s", self.client_id,
                     self.datasource.num_train_examples())
        
        ########################################################
        from plato.datasources.yolov5.utils.dataloaders import LoadImagesAndLabels
        single_class = (Config().data.num_classes == 1)
        self.datasource.trainset = LoadImagesAndLabels(
            Config().data.train_path,
            # self.image_size,  # 默认640
            Config().trainer.batch_size,
            augment=False,  # augment images
            hyp=None,  # augmentation hyperparameters
            rect=False,  # rectangular training
            cache_images=False,
            single_cls=single_class,
            # stride=int(self.grid_size),   # 默认32
            pad=0.0,
            image_weights=False,
            prefix=''
        )
        ########################################################

        # Setting up the data sampler
        logging.info("11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111 %s", type(self.datasource))
        logging.info("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww %s", type(self.datasource.trainset))
        self.sampler = samplers_registry.get(self.datasource, self.client_id)

        if hasattr(Config().trainer, 'use_mindspore'):
            # MindSpore requires samplers to be used while constructing
            # the dataset
            self.trainset = self.datasource.get_train_set(self.sampler)
        else:
            # PyTorch uses samplers when loading data with a data loader
            self.trainset = self.datasource.get_train_set()

        if Config().clients.do_test:
            # Set the testset if local testing is needed
            self.testset = self.datasource.get_test_set()

        self.data_loading_time = time.perf_counter() - data_loading_start_time

    def load_payload(self, server_payload) -> None:
        """Loading the server model onto this client."""
        self.algorithm.load_weights(server_payload)

    async def train(self):
        """The machine learning training workload on a client."""
        if hasattr(Config().clients, 'async_training') \
                and Config().clients.async_training is True:
            if not self.trainer.async_training_begun:
                logging.info("[Client #%d] Started training.", self.client_id)
                self.trainer.train(self.trainset, self.sampler)
                return None, None
        else:
            logging.info("[Client #%d] Started training.", self.client_id)

        # Perform model training
        try:
            training_time = self.trainer.train(self.trainset, self.sampler)
        except ValueError as e:
            logging.info("ValueError: %s", e)
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.trainer.test(self.testset)
            if accuracy == 0:
                # The testing process failed, disconnect from the server
                await self.sio.disconnect()

            model_name = Config().trainer.model_name
            if 'albert' in model_name:
                logging.info("[Client #{:d}] Test perplexity: {:.2f}".format(
                    self.client_id, accuracy))
            else:
                logging.info("[Client #{:d}] Test accuracy: {:.2f}%".format(
                    self.client_id, 100 * accuracy))
        else:
            accuracy = 0

        data_loading_time = 0

        if not self.data_loading_time_sent:
            data_loading_time = self.data_loading_time
            self.data_loading_time_sent = True

        if hasattr(Config().server, 'client_selection') \
                and Config().server.client_selection.name == 'oort' or \
                hasattr(Config().server, 'asynchronous') and \
                hasattr(Config().server.asynchronous, 'sirius'):
            report = OortReport(self.sampler.trainset_size(), accuracy,
                                training_time, data_loading_time,
                                self.trainer.get_utility())
        else:
            report = Report(self.sampler.trainset_size(), accuracy,
                            training_time, data_loading_time)
        return report, weights
