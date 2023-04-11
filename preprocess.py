from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset


class Preprocessor:
    def __init__(self, data_dfs, domains) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.block_size = 128
        self.preprocessed_data = {}

        for domain in domains:
            data_df = data_dfs[domain]
            abstract_files = {"train": data_df['train'].to_csv(), "test": data_df['test'].to_csv()}
            abstract_dataset = load_dataset("csv", data_files=abstract_files)
            tokenized_abstract = abstract_dataset.map(self.preprocess_function, batched=True, num_proc=4, remove_columns=["text", "Unnamed: 0"])
            lm_abstract = tokenized_abstract.map(self.group_texts,batched=True,batch_size=1000,num_proc=4)
            lm_abstract.set_format('torch', columns=["input_ids", "attention_mask", "labels"])
            self.preprocessed_data[domain] = lm_abstract

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding=True)
    
    def group_texts(self, examples):
        block_size = self.block_size
        # Concatenate all texts.
        # print(examples)
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    

