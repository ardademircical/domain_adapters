from transformers import Trainer, TrainingArguments
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """
    def __init__(self, batch_domain, data_loader):
        self.batch_domain = batch_domain
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)
    
    def __iter__(self):
        for batch in self.data_loader:
            # batch["batch_domain"] = self.batch_domain
            yield batch

  
class MultiDomainDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            batch_domain: len(dataloader) 
            for batch_domain, dataloader in self.dataloader_dict.items()
        }
        self.domain_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) 
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        domain_choice_list = []
        for i, domain_name in enumerate(self.domain_name_list):
            domain_choice_list += [i] * self.num_batches_dict[domain_name]
        domain_choice_list = np.array(domain_choice_list)
        np.random.shuffle(domain_choice_list)
        dataloader_iter_dict = {
            domain_name: iter(dataloader) 
            for domain_name, dataloader in self.dataloader_dict.items()
        }
        for domain_choice in domain_choice_list:
            domain_name = self.domain_name_list[domain_choice]
            yield next(dataloader_iter_dict[domain_name])    

class MultiDomainTrainer(Trainer):

    def get_single_train_dataloader(self, domain_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        else:
            train_sampler = (
                RandomSampler(train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(train_dataset)
            )

        data_loader = DataLoaderWithTaskname(
            batch_domain=domain_name,
            data_loader=DataLoader(
              train_dataset,
              batch_size=self.args.train_batch_size,
              sampler=train_sampler,
              collate_fn=self.data_collator,
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each 
        task Dataloader
        """
        return MultiDomainDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in self.train_dataset.items()
        })