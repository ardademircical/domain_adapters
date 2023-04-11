# S2ORC Initialization

import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd

class S2ORCInitializer:
   
    def __init__(self, meta_dir) -> None:
        self.METADATA_DIR = meta_dir
    
    def get_data(self, domains:list, sample_size:int, train_size:int):
        domain_paper_id_full = defaultdict(list)
        abstracts_per_domain = defaultdict(list)
        titled_abstracts_per_domain = defaultdict(list)
                
        all_meta = {}
        abstracts = {}
        pid_set = set()
        pid_to_domains = {}

        for metadata_file in os.listdir(self.METADATA_DIR):
            with open(os.path.join(self.METADATA_DIR, metadata_file)) as f_meta:
                for line in f_meta:
                    metadata_dict = json.loads(line)
                    domain = metadata_dict['mag_field_of_study']
                    pid = metadata_dict['paper_id']
                    if pid in pid_set or len(domain) > 1:
                        continue
                    pid_set.add(pid)
                    pid_to_domains[pid] = domain
                    title = metadata_dict['title']
                    abstract = metadata_dict['abstract']
                    if abstract == None:
                        continue
                    
                    for d in domain:
                        if d in domains and abstract != None and title != None:
                            abstracts_per_domain[d].append(abstract)
                            titled_abstracts_per_domain[d].append([title, abstract])
                   
        

        
        selected_titles_and_abstracts = defaultdict(list)
        batched_samples = []
        for key in titled_abstracts_per_domain:
            np.random.shuffle(titled_abstracts_per_domain[key])
            selected_titles_and_abstracts[key] = titled_abstracts_per_domain[key][:sample_size]
        selected_titles = defaultdict(list)
        selected_abstracts = defaultdict(list)

        for key in selected_titles_and_abstracts:
            for title, abstract in selected_titles_and_abstracts[key]:
                # selected_titles[key].append(title)    
                selected_abstracts[key].append(abstract)


        domain_abstracts_dfs = {}
        for domain in domains:
            domain_abstract = {"text": selected_abstracts[domain]}
            domain_abstract_df = pd.DataFrame.from_dict(domain_abstract)
            assert train_size <= len(domain_abstract_df)
            train_test_split = {}
            train_test_split['train'] = domain_abstract_df[:train_size]
            train_test_split['test'] = domain_abstract_df[train_size:]
            domain_abstracts_dfs[domain] = train_test_split

        return domain_abstracts_dfs
