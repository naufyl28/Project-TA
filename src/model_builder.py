import torch
import torch.nn as nn
from transformers import BertModel

class IndoBERTClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        use_hidden_layer: bool = False,
        hidden_size: int = 256,
        num_classes: int = 3,
        class_weights: torch.Tensor = None
    ):
        super(IndoBERTClassifier, self).__init__()
        
        # Load pretrained IndoBERT
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.use_hidden_layer = use_hidden_layer

        # Classifier architecture
        if use_hidden_layer:
            self.classifier = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, num_classes)
            )
        else:
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        # Loss function (CrossEntropy with class weights if provided)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        # Get BERT output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation

        # Classify
        logits = self.classifier(pooled_output)

        # Ensure labels are Long type (necessary for CrossEntropyLoss)
        if labels is not None:
            labels = labels.long()  # Convert labels to Long tensor

            # Compute loss
            loss = self.loss_fn(logits, labels)
            return {'loss': loss, 'logits': logits}
        
        return {'logits': logits}
