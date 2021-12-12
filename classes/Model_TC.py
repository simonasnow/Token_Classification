import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TokenClassifier(nn.Module):
    def __init__(self, model_name, num_labels=9, dropout_rate=0.1, freeze_bert=False):
        super(TokenClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.cls_size = int(config.hidden_size)
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(self.cls_size,num_labels)
        
        if freeze_bert:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        model_outputs = self.encoder(input_ids, attention_mask)
        last_hidden_state_cls = model_outputs[0] #contextual embedding of every token
        last_hidden_state_cls_dp = self.input_dropout(last_hidden_state_cls)
        logits = self.output_layer(last_hidden_state_cls_dp)
        return logits