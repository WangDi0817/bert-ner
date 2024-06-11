# from model import Crf
from crf1 import CRF1
from gen_data import Gen_data
import torch
import argparse
import numpy as np
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
from matplotlib import pyplot as plt
from util_ner import compute_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.train_dataset = Gen_data(args.data_file, "train", args.model_path, num_examples=4000)
        self.test_dataset = Gen_data(args.data_file, "test", args.model_path, num_examples=1000)
        self.tag_label = {value: key for key, value in self.train_dataset.label_map.items()}
        self.encoder = AutoModel.from_pretrained(args.model_path).cuda().eval()
        self.encoder.to(device)
        # self.load_model()
        self.model = CRF1(args.input_dim, args.hidden_dim, len(self.tag_label)+1, rate=args.rate)
        self.model.to(device)
        # self.loss_fn = CrossEntropyLoss()

    def train(self):
        train_dataloader = DataLoader(self.train_dataset, sampler=RandomSampler(self.train_dataset),
                                      batch_size=self.args.train_batch_size)

        # 参数初始化
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        # 训练步数
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.epochs = self.args.max_steps // (
                    len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.epochs

        # 定义优化器，调度器
        optimizer = AdamW(optimizer_params, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        global train_loss
        for epoch in range(self.args.epochs):
            global_step = 0
            tr_loss = 0.0
            self.model.train()
            for step, batch in enumerate(train_dataloader):  # batch:list(5)
                data = tuple(t.to(device) for t in batch[1:])  # 第一个是prompt，不跑
                # "prompt, input_ids, token_type_ids, attention_mask, label"
                inputs = {'input_ids': data[0],
                          'attention_mask': data[2],
                          'token_type_ids': data[1]}
                with torch.no_grad():
                    outputs = self.encoder(**inputs)[0]
                # [损失, 返回的结果张量(batch_size, max_length, num_labels)]
                logits = self.model(input_=outputs, labels=data[3], attention_mask=data[2])
                # loss = self.loss_fn(logits, data[3])
                loss = logits[0]
                loss.backward()
                tr_loss += loss.item()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                global_step += 1
            print("Epoch: {}/{} step: {}/{} Loss: {} AVG_Loss: {} Lr:{}".format(
                epoch + 1, self.args.epochs, step + 1, len(train_dataloader), loss.item(),
                tr_loss / global_step, optimizer.state_dict()['param_groups'][0]['lr']))
            train_loss.append(tr_loss / global_step)
            self.evaluate()
            self.save_model()
        return global_step, tr_loss / global_step

    def evaluate(self):
        eval_sampler = SequentialSampler(self.test_dataset)
        eval_dataloader = DataLoader(self.test_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        tag_preds = None
        tag_trues = None
        global eval_loss_list
        global eval_acc_list
        self.model.eval()
        for step, batch in enumerate(eval_dataloader):
            data = tuple(t.to(device) for t in batch[1:])  # GPU or CPU
            "prompt, input_ids, token_type_ids, attention_mask, label"
            inputs = {'input_ids': data[0],
                      'attention_mask': data[2],
                      'token_type_ids': data[1]}
            with torch.no_grad():
                outputs = self.encoder(**inputs)[0]
                logits = self.model(input_=outputs, labels=data[3], attention_mask=data[2])
                # logits = self.model(outputs)
                # loss = self.loss_fn(logits, data[3])
                loss = logits[0]
                pred = logits[1]
                eval_loss += loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if tag_preds is None:
                tag_preds = np.array(self.model.crf.decode(pred))
                tag_trues = batch[4].detach().cpu().numpy()

            else:
                tag_preds = np.append(tag_preds, np.array(self.model.crf.decode(pred)), axis=0)
                tag_trues = np.append(tag_trues, batch[4].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
        # results = {
        #     "loss": eval_loss
        # }

        # tag_preds = np.argmax(tag_preds, axis=1)  #
        acc = compute_metrics(tag_preds, tag_trues)
        # total_result = compute_metrics(tag_preds, tag_trues)
        # results.update(total_result)
        results = {
            "loss": eval_loss,
            "acc": acc,
        }
        print(f"avg eval loss : {eval_loss}, acc: {acc}")
        eval_loss_list.append(eval_loss)
        eval_acc_list.append(acc)
        # eval_loss_list.append(eval_loss)
        # eval_acc_list.append(acc)
        return results

    def save_model(self):
        filename = "ner.pth.tar"
        state = {"classify": self.model}
        torch.save(state, filename)

    def load_model(self):
        self.model = torch.load("ner.pth.tar")["classify"].to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kkkkkkkk')
    parser.add_argument('--data_file',
                        default="./common_tag_classifier.txt", help="data_file")
    parser.add_argument('--input_dim', type=int, default=768, help='dimension of word embeddings.')
    parser.add_argument('--hidden_dim', type=int, default=768, help='dimension of hidden linear layers.')
    parser.add_argument('--rate', type=float, default=0.2, help='dropout 比例')
    parser.add_argument('--num_labels', type=int, default=37, help='标签数量')
    parser.add_argument('--model_path', default=r"E:\LLM\pretrain_labse", help='预训练模型路径')
    parser.add_argument('--train_batch_size', type=int, default=32, help='train batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='train batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate for classify')
    parser.add_argument("--epochs", default=50, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    args = parser.parse_args()
    train_loss = []
    eval_loss_list = []
    eval_acc_list = []
    t1 = Trainer(args)
    t1.train()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_title('loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.plot(train_loss, 'r', label='train_loss')
    ax1.plot(eval_loss_list, 'b', label='eval_loss')
    ax1.legend()
    ax2.set_title('acc')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('acc')
    ax2.plot(eval_acc_list, 'b', label='eval_acc')
    ax2.legend()
    plt.show()
