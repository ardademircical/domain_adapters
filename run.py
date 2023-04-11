import os, sys
import argparse
from pathlib import Path
from multi_domain_trainer import MultiDomainDataloader, MultiDomainTrainer, DataLoaderWithTaskname
from distilbert_with_adapter import Distilbert_with_adapter
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from preprocess import Preprocessor
from s2orc_initializer import S2ORCInitializer
import torch



# example: python train_run.py keyword temp_keyword _
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='domain adapter training details.')
    parser.add_argument('--mode', type=str, default='base_adapter', help='')
    parser.add_argument('--domains', type=str, default='[]', help='')
    parser.add_argument('--single_domain', type=str, default=None, help='')
    parser.add_argument('--meta_dir', type=str, default=None, help='where you store your s2orc meta files')
    parser.add_argument('--output_dir', type=str, default=None, help='where you want to store your results')
    parser.add_argument('--sample_size', type=int, default=None, help='full sample size')
    parser.add_argument('--train_size', type=int, default=None, help='training sample size')


    args = parser.parse_args()
    
    domains = args.domains
    domains = domains.rstrip(domains[-1])

    domains = domains[1: ]
    domains = domains.split(", ")

    s2orc_initializer = S2ORCInitializer(args.meta_dir)
    data_dfs = s2orc_initializer.get_data(args.domains, args.sample_size, args.train_size)
    preprocessor = Preprocessor(data_dfs, args.domains)
    preprocessed_data = preprocessor.preprocessed_data
    tokenizer = preprocessor.tokenizer
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    train_dataset = {}
    eval_dataset = {}
    for domain in domains:
        train_dataset[domain] = preprocessed_data[domain]['train']
        eval_dataset[domain] = preprocessed_data[domain]['test']
        

    if args.mode == "base_adapter":
        model = Distilbert_with_adapter(domains=domains, single_domain=args.single_domain)
        
        training_args = TrainingArguments(
                output_dir=args.output_dir,
                overwrite_output_dir=True,
                learning_rate=1e-4,
                do_train=True,
                num_train_epochs=10,
                per_device_train_batch_size=8,  
                save_strategy ="no",
                eval_accumulation_steps = 15,
                fp16=True
                # save_steps=3000,
            )
        
        trainer = MultiDomainTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
    
        model.train()
        trainer.train()

        evals_dict = {}
        model.eval()
        for domain in domains:
            torch.cuda.empty_cache()
            torch.no_grad()
            model.eval()
            eval_dataloader = DataLoaderWithTaskname(
                domain,
                trainer.get_eval_dataloader(eval_dataset[domain]['test'])
            )
            evals_dict[domain] = trainer.evaluation_loop(
                eval_dataloader, 
                description=f"Validation: {domain}",
            )




