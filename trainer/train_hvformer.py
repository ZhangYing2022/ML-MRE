import torch
from torch import optim
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers.optimization import get_linear_schedule_with_warmup
import pandas as pd
from utilities.metrics import eval_result
import torch.nn.functional as F
import os
import numpy as np



class BertVitReTrainer(object):
    def __init__(self, train_data=None, dev_data=None, test_data=None, re_dict=None,
                 model=None, process=None,
                 args=None, logger=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.re_dict = re_dict
        self.model = model
        self.process = process
        self.logger = logger
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
        self.best_test_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
        self.final_test_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
        self.best_dev_epoch = None
        self.best_test_epoch = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args
        self.pbar = None
        self.re_optimizer = None
        self.re_scheduler = None
        self.best_threshold = 0.5  # 默认0.5
        self.before_train()

    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        # if self.args.load_path is not None:  # load model from load_path
        #     self.logger.info("Loading model from {}".format(self.args.load_path))
        #     self.model.load_state_dict(torch.load(self.args.load_path))
        #     self.logger.info("Load model successful!")

        if self.args.do_test:
            self.logger.info("***** Start testing without training *****")
            self.test(0)
            return

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            re_avg_loss = 0
            for epoch in range(1, self.args.num_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                visual_feature = []
                text_feature = []
                for batch in self.train_data:
                    self.step += 1
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (re_loss, re_logits, _),\
                        labels, _ = self._step(re_batch, mode="train", task='re', epoch=epoch)


                    re_avg_loss += re_loss.detach().cpu().item()
                    re_loss.backward()
                    self.re_optimizer.step()
                    self.re_optimizer.zero_grad()
                    self.re_scheduler.step()
                    if self.step % self.refresh_step == 0:
                        re_avg_loss = float(re_avg_loss) / self.refresh_step
                        print_output = "RE loss:{:<6.5f}".format(re_avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        re_avg_loss = 0
                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)
                    #self.test(epoch)

            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, "
                             "best dev f1 is {}".format(self.best_dev_epoch,
                                                        self.best_dev_metrics['micro_f1'],
                                                        ))
            self.logger.info(
                "Get best test performance at epoch {}, "
                "best test f1 is {}".format(self.best_test_epoch,
                                            self.best_test_metrics['micro_f1'],
                                            ))
            self.logger.info(
                "Get final test performance according to validation results at epoch {}, "
                "final f1 {}, "
                "recall {}, "
                "precision {}, "
                "acc {}".format(
                    self.best_dev_epoch,
                    self.final_test_metrics['micro_f1'],
                    self.final_test_metrics['micro_r'],
                    self.final_test_metrics['micro_p'],
                    self.final_test_metrics['acc']))
            self.logger.info(
                "Get best test performance at epoch {}, "
                "best test f1 {}, "
                "recall {}, "
                "precision {}, "
                "acc {}".format(
                    self.best_test_epoch,
                    self.best_test_metrics['micro_f1'],
                    self.best_test_metrics['micro_r'],
                    self.best_test_metrics['micro_p'],
                    self.best_test_metrics['acc']))

    def evaluate(self, epoch=0):
        self.model.eval()
        self.logger.info(f"***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                step = 0
                all_true = []
                all_probs = []
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (loss, logits, _), labels, _ = self._step(re_batch,
                                                          mode="dev",
                                                          task='re',
                                                          epoch=epoch)
                    total_loss += loss.detach().cpu().item()

                    # 多标签预测：sigmoid+阈值
                    probs = torch.sigmoid(logits)
                    all_true.append(labels.detach().cpu())
                    all_probs.append(probs.detach().cpu())
                # 拼接所有batch
                all_true = torch.cat(all_true, dim=0).numpy()
                all_probs = torch.cat(all_probs, dim=0).numpy()

                # # 搜索最佳阈值
                # best_thresh, best_f1 = self.find_best_threshold(all_true, all_probs, self.re_dict)
                # self.logger.info(f"Best threshold on dev: {best_thresh:.3f}, best micro_f1: {best_f1:.4f}")

                # 用最佳阈值做最终评估
                all_pred = (all_probs > 0.5).astype(int)

                # 计算多标签指标
                report = classification_report(
                    all_true, all_pred,
                    target_names=list(self.re_dict.keys()),
                    digits=4,
                    zero_division=0
                )
                micro_f1  = eval_result(all_true, all_pred, rel2id=self.re_dict)['micro_f1']
                micro_p = eval_result(all_true, all_pred, rel2id=self.re_dict)['micro_p']
                micro_r = eval_result(all_true, all_pred, rel2id=self.re_dict)['micro_r']
                acc = eval_result(all_true, all_pred, rel2id=self.re_dict)['acc']
                macro_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)

                self.logger.info("\n" + report)
                self.logger.info(
                    f"Epoch {epoch}/{self.args.num_epochs}, micro_f1: {micro_f1:.4f}, macro_f1: {macro_f1:.4f}, "
                    f"micro_p: {micro_p:.4f}, micro_r: {micro_r:.4f}, acc: {acc:.4f}"
                )
                print(f"Epoch {epoch}/{self.args.num_epochs}, micro_f1: {micro_f1:.4f}, macro_f1: {macro_f1:.4f}, "
                    f"micro_p: {micro_p:.4f}, micro_r: {micro_r:.4f}, acc: {acc:.4f}")
                
                # 更新best指标
                if micro_f1 >= self.best_dev_metrics['micro_f1']:
                    self.logger.info("Get better dev performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metrics['micro_f1'] = micro_f1
                    self.best_dev_metrics['micro_r'] = micro_r
                    self.best_dev_metrics['micro_p'] = micro_p
                    self.best_dev_metrics['acc'] = acc
                    if self.args.save_path is not None:
                        save_dir = os.path.dirname(self.args.save_path)
                        if save_dir and not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        torch.save(self.model.state_dict(), self.args.save_path)
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def test(self, epoch=0):
        self.model.eval()
        self.logger.info(f"\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        all_true = []
        all_pred = []
        sample_word_lists, sample_image_ids = [], []
        re_pred_logits = []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    if self.args.write_path is not None and self.args.do_test:
                        (loss, logits, _), labels, extend_word_lists, imgids = self._step(
                            re_batch, mode="test", task='re', epoch=epoch)
                    else:
                        (loss, logits, _), labels, _ = self._step(
                            re_batch, mode="test", task='re', epoch=epoch)
                    total_loss += loss.detach().cpu().item()

                    # 多标签预测
                    probs = torch.sigmoid(logits)
                    preds = (probs > self.best_threshold).long()

                    all_true.append(labels.detach().cpu())
                    all_pred.append(preds.detach().cpu())
                    re_pred_logits.extend(logits.detach().cpu().tolist())
                    if self.args.write_path is not None and self.args.do_test:
                        sample_word_lists.extend([*extend_word_lists])
                        sample_image_ids.extend([*imgids])
                    pbar.update()
                pbar.close()

                # 拼接所有batch
                all_true = torch.cat(all_true, dim=0).numpy()
                all_pred = torch.cat(all_pred, dim=0).numpy()

                # 多标签指标
                report = classification_report(
                    all_true, all_pred,
                    target_names=list(self.re_dict.keys()),
                    digits=4,
                    zero_division=0
                )
                micro_f1  = eval_result(all_true, all_pred, rel2id=self.re_dict)['micro_f1']
                micro_p = eval_result(all_true, all_pred, rel2id=self.re_dict)['micro_p']
                micro_r = eval_result(all_true, all_pred, rel2id=self.re_dict)['micro_r']
                acc = eval_result(all_true, all_pred, rel2id=self.re_dict)['acc']
                macro_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
                print(f"Epoch {epoch}/{self.args.num_epochs}, micro_f1: {micro_f1:.4f}, macro_f1: {macro_f1:.4f}, "
                    f"micro_p: {micro_p:.4f}, micro_r: {micro_r:.4f}, acc: {acc:.4f}")

                # if self.args.write_path is not None and self.args.do_test:
                #     write_file_dict = {
                #         'sample_word_lists': sample_word_lists,
                #         'sample_image_ids': sample_image_ids,
                #         'true_labels': all_true.tolist(),
                #         'pred_labels': all_pred.tolist(),
                #         'pred_logits': re_pred_logits
                #     }
                #     df = pd.DataFrame(write_file_dict)
                #     df.to_csv(self.args.write_path + '_' + 'test.csv')

                self.logger.info("\n" + report)
                self.logger.info(
                    f"Epoch {epoch}/{self.args.num_epochs}, micro_f1: {micro_f1:.4f}, macro_f1: {macro_f1:.4f}, "
                    f"micro_p: {micro_p:.4f}, micro_r: {micro_r:.4f}, acc: {acc:.4f}"
                )

                # 更新best指标
                if micro_f1 > self.final_test_metrics['micro_f1']:
                    self.final_test_metrics['micro_f1'] = micro_f1
                    self.final_test_metrics['micro_r'] = micro_r
                    self.final_test_metrics['micro_p'] = micro_p
                    self.final_test_metrics['acc'] = acc

                if micro_f1 >= self.best_test_metrics['micro_f1']:
                    self.logger.info("Get better test performance at epoch {}".format(epoch))
                    self.best_test_epoch = epoch
                    self.best_test_metrics['micro_f1'] = micro_f1
                    self.best_test_metrics['micro_r'] = micro_r
                    self.best_test_metrics['micro_p'] = micro_p
                    self.best_test_metrics['acc'] = acc

        self.model.train()

    def predict(self, dataloaders, ckpt_path):
        self.model.eval()
        data = dataloaders
        self.logger.info(f"\n***** Running prediction *****")
        self.logger.info("  Num instance = %d", len(data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        if ckpt_path is not None:
            self.logger.info("Loading model from {}".format(ckpt_path))
            self.model.load_state_dict(torch.load(ckpt_path))
            self.logger.info("Load model successful!")
        re_true_labels, re_pred_labels, sample_word_lists, sample_image_ids = [], [], [], []
        re_pred_logits = []

        with torch.no_grad():
            with tqdm(total=len(data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Predicting")
                mm_logits = []
                m1_logits = []
                m2_logits = []
                m1_feature = []
                m2_feature = []
                for i, batch in enumerate(data):
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                                batch)
                    (_, logits, output_for_contribution), labels, _ = self._step(re_batch,
                                                               mode="test",
                                                               task='re',
                                                               epoch=0,)
                    logits_v = output_for_contribution[0]
                    logits_t = output_for_contribution[1]
                    m1_feature.append(output_for_contribution[2])
                    m2_feature.append(output_for_contribution[3])
                    mm_logits.append(logits)
                    m1_logits.append(logits_v)
                    m2_logits.append(logits_t)
                    #torch.save(attention, f'entity_attentions_for_batch{i}.pt'.format(i))
                    pbar.update()

                mm_logits = torch.cat(mm_logits, dim=0)
                mm_probs = torch.softmax(mm_logits, dim=-1)
                m1_feature = torch.cat(m1_feature, dim=0)
                m2_feature = torch.cat(m2_feature, dim=0)

                mm_logits_drop_m1 = torch.cat(m2_logits, dim=0)
                mm_logits_drop_m2 = torch.cat(m1_logits, dim=0)

                mm_probs_m1_drop = torch.softmax(mm_logits_drop_m1, dim=-1)
                mm_probs_m2_drop = torch.softmax(mm_logits_drop_m2, dim=-1)
                #mm_probs_m1_m2_drop = torch.softmax(mm_logits_drop_m1_m2, dim=-1)

                # if delta_m1 is small, meaning that m1 contributes less
                delta_m1 = (mm_probs - mm_probs_m1_drop + mm_probs_m2_drop) / 2
                delta_m2 = (mm_probs - mm_probs_m2_drop + mm_probs_m1_drop) / 2

                delta_m1_logits = (mm_logits - mm_logits_drop_m1 + mm_logits_drop_m2) / 2
                delta_m2_logits = (mm_logits - mm_logits_drop_m2 + mm_logits_drop_m1) / 2

                contribution_m1 = torch.abs(delta_m1) / (torch.abs(delta_m1) + torch.abs(delta_m2))
                contribution_m2 = torch.abs(delta_m2) / (torch.abs(delta_m1) + torch.abs(delta_m2))
                pbar.close()
        return mm_probs, m1_feature, m2_feature, contribution_m1, contribution_m2

    def find_best_threshold(self, y_true, y_probs, re_dict):
        """
        在验证集上搜索最佳阈值（针对micro F1）
        y_true: numpy array, shape (num_samples, num_labels)
        y_probs: numpy array, shape (num_samples, num_labels)
        """
        best_threshold = 0.5
        best_f1 = 0
        thresholds = np.arange(0.1, 0.9, 0.01)
        for thresh in thresholds:
            y_pred = (y_probs > thresh).astype(int)
            f1 = eval_result(y_true, y_pred, rel2id=re_dict)['micro_f1']
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        return best_threshold, best_f1


    def _step(self, batch, mode="train", task='re', epoch=0):
        if self.args.write_path is not None and mode == 'test' and self.args.do_test:
            input_ids, token_type_ids, attention_mask, labels, images = batch
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 labels=labels,
                                 images=images,
                                 aux_imgs=None,
                                 rcnn_imgs=None,
                                 task=task,
                                 epoch=epoch,)
            return outputs, labels
        else:
            if task == 're':
            # if task == 're':
            #     input_ids = re_input_ids
            #     token_type_ids = re_token_type_ids
            #     attention_mask = re_attention_mask
            #     labels = re_labels
                input_ids, token_type_ids, attention_mask, labels, images = batch
                outputs = self.model(input_ids=input_ids,attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     labels=labels,images=images,
                                     aux_imgs=None
                                     ,rcnn_imgs=None,
                                     task=task, epoch=epoch)


            return outputs, labels, attention_mask

    def before_train(self):
        optimizer_grouped_parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if 'model' in name or name.startswith('re_classifier'):
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)
        self.re_optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.re_scheduler = get_linear_schedule_with_warmup(optimizer=self.re_optimizer,
                                                            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                            num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)


