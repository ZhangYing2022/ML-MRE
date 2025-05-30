import os
import argparse
import logging
import sys
sys.path.append("..")

import torch
import numpy as np
import random
#from models import get_distillation_kernel_homo
from torchvision import transforms
from torch.utils.data import DataLoader
from models.bert_model import HvPNeTREModel, HvPNeTNERModel
from processor.dataset_wophrase import MMREProcessor, MMPNERProcessor, MMREDataset, MMPNERDataset
from trainer.train_hvpnet import RETrainer, NERTrainer
import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
warnings.filterwarnings("ignore", category=UserWarning)
# from tensorboardX import SummaryWriter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'MRE': HvPNeTREModel,
}

TRAINER_CLASSES = {
    'MRE': RETrainer,
}
DATA_PROCESS = {
    'MRE': (MMREProcessor, MMREDataset),
}

DATA_PATH = {
    'MRE': {
            # text data
        'train': 'train.json',
         'dev': 'val.json',
         'test': 'test.json', 
            # relation json data
            're_path': 'label2id.json'
            },

            },
        
}

# image data
IMG_PATH = {
    'MRE': {'train': 'mmdata/',
            'dev': 'mmdata/',
            'test': 'mmdata/',
            }

}

# auxiliary images
AUX_PATH = {
    'MRE':{
            'train': None,
            'dev': None,
            'test': None
    }
}

def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def softmax(w, t=1.0, axis=None):
  w = np.array(w) / t
  e = np.exp(w - np.amax(w, axis=axis, keepdims=True))
  dist = e / np.sum(e, axis=axis, keepdims=True)
  return dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='MRE', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="Pretrained language model path")
    parser.add_argument('--num_epochs', default=15, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=8, type=int, help="batch size")
    parser.add_argument('--lr', default=3e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.06, type=float)
    parser.add_argument('--eval_begin_epoch', default=1, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=1234, type=int, help="random seed, default is 1")
    parser.add_argument('--prompt_len', default=4, type=int, help="prompt length")
    parser.add_argument('--prompt_dim', default=800, type=int, help="mid dimension of prompt project layer")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default='ckpt/re/', type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--use_prompt', action='store_false')
    parser.add_argument('--do_train', action='store_false')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--sample_ratio', default=1, type=float, help="only for low resource.")

    args = parser.parse_args()

    data_path, img_path, aux_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[args.dataset_name]
    model_class, Trainer = MODEL_CLASSES[args.dataset_name], TRAINER_CLASSES[args.dataset_name]
    data_process, dataset_class = DATA_PROCESS[args.dataset_name]


    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    set_seed(args.seed) # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        # args.save_path = os.path.join(args.save_path, args.dataset_name+"_"+str(args.batch_size)+"_"+str(args.lr)+"_"+args.notes)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)
    logdir = "logs/" + args.dataset_name+ "_"+str(args.batch_size) + "_" + str(args.lr) + args.notes
    # writer = SummaryWriter(logdir=logdir)
    writer=None

    if not args.use_prompt:
        img_path, aux_path = None, None

        
    processor = data_process(data_path, args.bert_name)
    train_dataset = dataset_class(processor, transform, img_path, aux_path,  args.max_seq, sample_ratio=args.sample_ratio, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    dev_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.dataset_name == 'MRE':  # RE task
        re_dict = processor.get_relation_dict()
        num_labels = len(re_dict)
        tokenizer = processor.tokenizer
        model = HvPNeTREModel(num_labels, tokenizer, args=args)
        #model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                          processor=processor, args=args, logger=logger, writer=writer)


    if args.do_train:
        # train
        trainer.train()
        # test best model
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        trainer.test()

    if args.only_test:
        # only do test
        trainer.test()

    torch.cuda.empty_cache()
    # writer.close()


if __name__ == "__main__":
    main()