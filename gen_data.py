import random
import torch

from transformers import AutoTokenizer, AutoModel


# all_labels = ["质量", "渲染", "类型", "IP", "流派", "艺术家", "镜头", "色彩", "数量", "身份", "动漫", "五官",
#               "表情", "胸部", "四肢", "特征", "头部", "基础", "复杂", "饰品", "套装", "上装", "下装", "鞋/袜",
#               "家具", "载具", "道具", "食物", "植物", "动物", "天气", "室内", "室外", "建筑", "氛围", "天体"]
# label_map = {all_labels[i]: i for i in range(len(all_labels))}
#
# encoder = AutoModel.from_pretrained('../pretrain_labse').cuda().eval()
# prompt, label = joint(prompt_zh, 100)
# prompt_id = []
# label_id = []
# tokenizer = AutoTokenizer.from_pretrained('../pretrain_labse')
# for i in range(len(prompt)):
#     token_id = tokenizer(prompt[i])
#     prompt_id.append(token_id)
#     label_id.append([label_map[j] for j in label[i]])
# input_ids, token_type_ids, attention_masks = [], [], []
# pad_token_id = tokenizer.pad_token_id
# for tokens in prompt_id:
#     padlength = 100 - len(tokens["input_ids"])
#     input_ids.append(tokens["input_ids"] + [pad_token_id] * padlength)
#     token_type_ids.append(tokens["token_type_ids"] + [0] * padlength)
#     attention_masks.append(tokens["attention_mask"] + [0] * padlength)
# input_ids, token_type_ids, attention_masks = torch.LongTensor(input_ids), torch.LongTensor(
#     token_type_ids), torch.LongTensor(attention_masks)
# inputs = {'input_ids': input_ids.to('cuda'),
#           'attention_mask': attention_masks.to('cuda'),
#           'token_type_ids': token_type_ids.to('cuda')}
# with torch.no_grad():
#     outputs = encoder(**inputs)
# print(1)
#

class Gen_data(object):
    def __init__(self, file_path, mode, model_path, num_examples):
        self.mode = mode
        self.num_examples = num_examples
        self.file = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.unk_token = self.tokenizer.unk_token
        self.pad_token_id = self.tokenizer.pad_token_id
        # self.start_id = self.tokenizer.bos_token_id
        # self.end_id = self.tokenizer.eos_token_id
        # 输出：str, list[tensor], list[tensor], list[tensor], list[tensor]
        self.prompt = []
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_id = []
        self.label_pad_id = 36

        self.max_length = 100
        all_labels = ["质量", "渲染", "类型", "IP", "流派", "艺术家", "镜头", "色彩", "数量", "身份", "动漫", "五官",
                      "表情", "胸部", "四肢", "特征", "头部", "基础", "复杂", "饰品", "套装", "上装", "下装", "鞋/袜",
                      "家具", "载具", "道具", "食物", "植物", "动物", "天气", "室内", "室外", "建筑", "氛围", "天体"]
        self.label_map = {all_labels[i]: i for i in range(len(all_labels))}
        self.process()

    def joint(self, text: list[list], num: int):
        # 标点比例
        punctuation = '，' * 45 + '。' * 5 + ',' * 45 + '.' * 5
        for i in range(int(num)):
            sample_zh = random.sample(text, 20)
            data = ''
            tokens = []
            slot_id = []
            # 分词， 对应label
            for word, slot_label in sample_zh:
                pun = random.choices(punctuation)[0]
                # 分词并添加
                word_tokens = self.tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [self.unk_token]
                tokens.extend(word_tokens)
                slot_id.extend([self.label_map[slot_label]] * len(word_tokens))
                data += word
                # 添加标点
                pun_tokens = pun
                tokens.append(pun_tokens)
                slot_id.append(self.label_pad_id)
                data += pun
            # 添加cls， sep前截断
            if len(tokens) > self.max_length - 2:
                tokens = tokens[:(self.max_length - 2)]
                slot_id = slot_id[:(self.max_length - 2)]
            # 添加sep
            tokens += [self.sep_token]
            slot_id += [self.label_pad_id]
            # 添加cls
            tokens = [self.cls_token] + tokens
            slot_id = [self.label_pad_id] + slot_id
            # 定义token_type_ids， input_ids， attention_mask
            token_type_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            # pad
            padding_length = self.max_length - len(input_ids)
            input_ids = torch.LongTensor(input_ids + [self.pad_token_id] * padding_length)
            attention_mask = torch.LongTensor(attention_mask + [0] * padding_length)
            token_type_ids = torch.LongTensor(token_type_ids + [0] * padding_length)
            slot_id = torch.LongTensor(slot_id + [self.label_pad_id] * padding_length)
            # 添加到self
            self.prompt.append(data[:-1])
            self.input_ids.append(input_ids)
            self.token_type_ids.append(token_type_ids)
            self.attention_mask.append(attention_mask)
            self.label_id.append(slot_id)

    def process(self):
        with open(self.file, 'r', encoding='utf_8') as f:
            lines = f.readlines()
            prompt_temp = []
            for i, line in enumerate(lines):
                temp = line.split('\t')
                prompt_temp.append([temp[0], temp[1][:-1]])  # [[prompt, label], ...]
            self.joint(prompt_temp, self.num_examples)

    def __getitem__(self, i):
        return self.prompt[i], self.input_ids[i], self.token_type_ids[i], self.attention_mask[i], self.label_id[i]

    def __len__(self):
        return len(self.prompt)


if __name__ == '__main__':
    "file_path, mode, model_path, num_examples):"
    g = Gen_data('common_tag_classifier.txt','train', r'E:\LLM\pretrain_labse',10)
    x = g.__getitem__(1)
    print(1)

