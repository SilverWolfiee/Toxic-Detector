import torch.nn as nn
from transformers import AutoModel

class TinyBERTToxicityModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
       
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  

        self.regressor = nn.Sequential(
            nn.Dropout(0.3),       
            nn.Linear(hidden, 128), 
            nn.ReLU(),            
            nn.Linear(128, 1),     
            nn.Sigmoid()          
        )
        self.identity_classifier = nn.Sequential(
            nn.Dropout(0.3),      
            nn.Linear(hidden, 1), 
            nn.Sigmoid()          
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           output_attentions=True)
        cls_emb    = out.last_hidden_state[:, 0, :]
        toxicity   = self.regressor(cls_emb).squeeze(-1)
        identity   = self.identity_classifier(cls_emb).squeeze(-1)
        attentions = out.attentions
        return toxicity, identity, attentions