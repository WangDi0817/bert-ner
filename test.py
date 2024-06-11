# -*- coding: utf-8 -*-
"""
@Time : 2023/3/17 14:49
@Auth : hushengyuan
@File : test_labse.py
@desc :
"""
import torch
import argparse
import json
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup



class Predict(object):
    def __init__(self, args):
        self.args = args
        # all_labels = ["角色", "眼睛", "身体", "头发", "表情", "动作", "服装", "场景", "画风", "构图", "物品"]
        # all_labels = ["描述", "画风", "画质"]
        # all_labels = ["0", "1"]
        all_labels = ["质量", "渲染", "类型", "IP", "流派", "艺术家", "镜头", "色彩", "数量", "身份", "动漫", "五官",
                                   "表情", "胸部", "四肢","特征", "头部", "基础", "复杂", "饰品", "套装", "上装", "下装", "鞋/袜",
                                   "家具", "载具", "道具", "食物", "植物", "动物", "天气", "室内", "室外", "建筑", "氛围", "天体"
        ]
        self.tag_label_dict = {i: all_labels[i] for i in range(len(all_labels))}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.encoder = AutoModel.from_pretrained(args.model_path).cuda().eval()
        self.load_model()
        self.model.to(self.device).eval()
        self.encoder.to(self.device).eval()
        self.max_length = 100
        self.pad_token_id = self.tokenizer.pad_token_id
        self.start_id = self.tokenizer.bos_token_id
        self.end_id = self.tokenizer.eos_token_id

    def predict(self, prompts):
        token_ids = [self.tokenizer(prompt) for prompt in prompts]
        input_ids, token_type_ids, attention_masks = [], [], []
        for tokens in token_ids:
            padlength = self.max_length - len(tokens['input_ids'])
            if padlength < 0:
                input_ids = torch.LongTensor(tokens['input_ids'][:100])
                token_type_ids = torch.LongTensor(tokens["token_type_ids"][:100])
                attention_masks = torch.LongTensor(tokens["attention_mask"][:100])

            else:
                input_ids = torch.LongTensor(tokens['input_ids'] + [self.pad_token_id] * padlength)
                token_type_ids = torch.LongTensor(tokens["token_type_ids"] + [0] * padlength)
                attention_masks = torch.LongTensor(tokens["attention_mask"] + [0] * padlength)
                # 特殊字符用-1作为label

        # input_ids, token_type_ids, attention_masks = torch.LongTensor(input_ids), torch.LongTensor(
        #     token_type_ids), torch.LongTensor(attention_masks)
        input_ids, token_type_ids, attention_masks = torch.unsqueeze(input_ids,0), \
            torch.unsqueeze(token_type_ids,0), torch.unsqueeze(attention_masks,0)
        "prompt, input_ids, token_type_ids, attention_mask, label"
        inputs = {'input_ids': input_ids.to(self.device),
                  'attention_mask': attention_masks.to(self.device),
                  'token_type_ids': token_type_ids.to(self.device)}
        with torch.no_grad():
            outputs = self.encoder(**inputs)[0]
            # outputs = self.encoder(**inputs)[0]
            logits = self.model(outputs)
            predicts = torch.nn.Softmax(dim=1)(logits)
            max_prob, predict = torch.max(predicts, 1)
            result = [(prompts[i], self.tag_label_dict[int(predict.cpu()[i])], max_prob.cpu()[i].item()) for i in
                      range(len(prompts))]

        # result = self.prepare_result(result)
        return result

    def save_model(self):
        filename = "common_classify.pth.tar"
        state = {"classify": self.model}
        torch.save(state, filename)

    def load_model(self):
        self.model = torch.load("ner.pth.tar")["classify"].to(self.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='labse classify')
    # Data parameters
    parser.add_argument('--prompt_file', default='common_tag_classifier.txt', help='path to prompt file')
    # parser.add_argument('--prompt_file', default='all_prompt_4w.txt',help='path to prompt file')
    parser.add_argument('--output_file', default='../result.txt', help='path to output file')
    parser.add_argument('--input_dim', type=int, default=768, help='dimension of word embeddings.')
    parser.add_argument('--hidden_dim', type=int, default=768, help='dimension of hidden linear layers.')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--model_path', default=r"E:\LLM\pretrain_labse", help='pretrained labse model path')
    parser.add_argument('--train_batch_size', type=int, default=32, help='train batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='train batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for classify')
    parser.add_argument("--epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    args = parser.parse_args()
    predictor = Predict(args)
    print(predictor.predict(['1girl', '(best quality:1.5)','standing on street', 'cowboy shot', 'blush', 'smile', 'wolf ears', 'wolf tail']))


    # results = []
    #
    # for each in prompts:
    #
    #     results.append(predictor.predict([each])[0])
    #
    # count = 0
    # for prompt, tag_pred, pro in results:
    #     tag_true = prompt.split('\t')[1]
    #     if tag_true == tag_pred:
    #         count += 1
    # accuracy = count/len(results)
    # print(accuracy)
    #
    # # write results to file
    # with open(args.output_file, 'w',encoding="utf-8") as f:
    #     for prompt, tag, accuracy in results:
    #         f.write(f'{prompt}\t{tag}\t{accuracy}\n')
    #
    #     f.write(f'准确率：{accuracy}')


