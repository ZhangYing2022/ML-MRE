import torch.cuda
from torch import nn, Tensor, device
#from torchcrf import CRF
import torch
import torch.nn.functional as F

from .bert_vit_inter_base_model import BertVitInterBaseModel, CLIPVisionEmbeddings, Vision2TextAttention, CLIPEncoderLayer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertModel, BertConfig, BertTokenizer, CLIPProcessor, CLIPModel


class REClassifier(nn.Module):
    def __init__(self, re_label_mapping=None, config=None, tokenizer=None):
        super().__init__()
        self.text_config = config
        num_relation_labels = len(re_label_mapping)
        self.classifier = nn.Linear(2 * self.text_config.hidden_size, num_relation_labels)

        self.head_start = tokenizer.convert_tokens_to_ids("<s>")  # <s> id: 30522
        self.tail_start = tokenizer.convert_tokens_to_ids("<e>")  # <o> id: 30526

    def forward(self, input_ids=None, output_state=None, output_attention=None, mode=None):
       
        (output_state, vision_hidden_states, text_hidden_states) = output_state
        last_hidden_state, pooler_output = output_state.last_hidden_state, output_state.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape

        entity_hidden_state = torch.Tensor(bsz, 2 * hidden_size)  # batch, 2*hidden
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)

        if torch.cuda.is_available():
            entity_hidden_state = entity_hidden_state.to('cuda')
        logits = self.classifier(entity_hidden_state)

        return logits, entity_hidden_state


# Bert VIT
class BertVitInterReModel(nn.Module):
    def __init__(self,
                 re_label_mapping=None,
                 tokenizer=None,
                 args=None,
                 vision_config=None,
                 text_config=None,
                 clip_model_dict=None,
                 bert_model_dict=None, ):
        super().__init__()
        self.args = args
        print(vision_config)
        print(text_config)
        self.vision_config = vision_config
        self.text_config = text_config
        vision_config.device = args.device
        self.model = BertVitInterBaseModel(vision_config, text_config, args)

        # test load:
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            if 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        self.model.load_state_dict(model_dict)
        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args
        # RE classifier
        self.re_classifier = REClassifier(re_label_mapping=re_label_mapping, config=text_config,
                                          tokenizer=tokenizer)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
            task='re',
            epoch=0,
    ):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            pixel_values=images,
                            aux_values=aux_imgs,
                            rcnn_values=rcnn_imgs,
                            output_attentions=False,
                            return_dict=True,
                            output_hidden_states=True,
                            )
        if task == 're':
            (output_state, vision_hidden_states, text_hidden_states) = output
            output_attention = output_state.attentions

            bert_vit_logits, entity_hidden_state = self.re_classifier(output_state=output, input_ids=input_ids, output_attention=output_attention, mode='bert_vit')
            # bert_vit_logits = self.re_classifier(output_state=output, input_ids=input_ids)
            if labels is not None:
                label_loss_bert_vit = F.binary_cross_entropy_with_logits(bert_vit_logits, labels.float())

                return label_loss_bert_vit, bert_vit_logits, entity_hidden_state


class BertForMultiLabelEntityClassification(nn.Module):
    def __init__(self,
                 re_label_mapping=None,
                 tokenizer=None,
                 args=None,
                 vision_config=None,
                 text_config=None,
                 clip_model_dict=None,
                 bert_model_dict=None, ):
        super().__init__()
        self.bert = BertModel.from_pretrained('/aistudio/workspace/cfs/xyz-data-tj/user/work/zhangying/dataset/bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, len(re_label_mapping))
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")  # <s> id: 30522
        self.tail_start = tokenizer.convert_tokens_to_ids("<e>")  # <o> id: 30526
        self.bert.resize_token_embeddings(len(tokenizer))
    def forward(self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
            task='re',
            epoch=0):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
        batch_size = input_ids.size(0)
        entity_hidden = []
        for i in range(batch_size):
            head_idx = (input_ids[i] == self.head_start).nonzero(as_tuple=True)[0]
            tail_idx = (input_ids[i] == self.tail_start).nonzero(as_tuple=True)[0]
            head_vec = last_hidden_state[i, head_idx, :].squeeze(0)
            tail_vec = last_hidden_state[i, tail_idx, :].squeeze(0)
            entity_hidden.append(torch.cat([head_vec, tail_vec], dim=-1))
        entity_hidden = torch.stack(entity_hidden, dim=0)  # [batch, hidden*2]
        logits = self.classifier(entity_hidden)
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            return loss, logits, entity_hidden
        else:
            return None, logits, entity_hidden





