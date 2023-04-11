from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM
from torch import nn

class DomainController(nn.Module):
  def __init__(self, current_batch_domain=None):
        super().__init__()
        self.current_batch_domain = current_batch_domain
  
class DomainAdapter(nn.Module):
    """
    The adapters first project the original
    d-dimensional features into a smaller dimension, m, apply
    a nonlinearity, then project back to d dimensions.
    """
    def __init__(self, size = 6, model_dim = 768):
        super().__init__()
        self.debugger = 1
        self.adapter_block = nn.Sequential(
            nn.Linear(model_dim, size),
            nn.ReLU(),
            nn.Linear(size, model_dim)
        )

    def forward(self, x):
        
        ff_out = self.adapter_block(x)
        adapter_out = ff_out + x
        return adapter_out


class Adapter(nn.Module):
    """
    The adapters first project the original
    d-dimensional features into a smaller dimension, m, apply
    a nonlinearity, then project back to d dimensions.
    """
    def __init__(self, size = 6, model_dim = 768):
        super().__init__()
        self.adapter_block = nn.Sequential(
            nn.Linear(model_dim, size),
            nn.ReLU(),
            nn.Linear(size, model_dim)
        )

    def forward(self, x):
        ff_out = self.adapter_block(x)
        adapter_out = ff_out + x
        return adapter_out

class Adaptered(nn.Module):
    def __init__(self, orig_layer, domains=[], domain_controller=None, no_base=False):
        super().__init__()
        self.orig_layer = orig_layer
        self.adapter = Adapter()
        self.domains = domains
        self.domain_controller = domain_controller
        self.no_base = no_base
        if not self.no_base:
          self.domain_adapters = nn.ModuleDict({domain: DomainAdapter() for domain in self.domains})

    def forward(self, *x):
        orig_out = self.orig_layer(*x)
        base_output = self.adapter.forward(orig_out)
        if self.domain_controller.current_batch_domain and not self.no_base:
          current_domain_adapter = self.domain_adapters[self.domain_controller.current_batch_domain]
          domain_output = current_domain_adapter.forward(base_output)
          return domain_output
        else:
          return base_output

class Distilbert_with_adapter(nn.Module):

    def __init__(self, domains=[], single_domain=None, no_base=False):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
    "distilbert-base-uncased", )
        self.domain_controller = DomainController()
        self.domains = domains
        self.no_base = no_base
        self.single_domain = single_domain
        if single_domain:
          self.domain_controller.current_batch_domain = single_domain
        # Freeze the original model parameters
        for params in self.model.parameters():
            params.requires_grad = False
        # Embed adapter layers into the transformer blocks 
        for i in range(6):
            self.model.distilbert.transformer.layer[i].ffn = Adaptered(self.model.distilbert.transformer.layer[i].ffn, domains, self.domain_controller)

    def get_model(self):
        return self.model

    def forward(self, input_ids, attention_mask, labels=None, batch_domain=None):
        if batch_domain and not self.single_domain:
          assert batch_domain in self.domains
          self.domain_controller.current_batch_domain = batch_domain
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)