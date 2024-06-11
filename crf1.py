import torch.nn as nn
from torchcrf import CRF


class CRF1(nn.Module):
    def __init__(self, input_dim, hidden_dim, tag_dim, rate):
        super(CRF1, self).__init__()
        # self.bert = BertModel(config)
        self.dropout_1 = nn.Dropout(rate)
        self.dropout_2 = nn.Dropout(rate)
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, tag_dim)
        # self.crf = CRF(num_tags=tag_dim, batch_first=True)
        self.crf = CRF(num_tags=tag_dim,batch_first=True)

        # self.init_weights()

    def forward(self, input_, labels, attention_mask):
        # outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        # sequence_output = outputs[0]
        output = self.dropout_1(input_)
        output = self.linear_1(output)
        output = self.dropout_2(output)
        logits = self.linear_2(output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask.byte(), reduction='mean')
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores
