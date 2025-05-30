import argparse
import pathlib
import sys
import time
import traceback
import warnings
import numpy as np
from transformers import BertConfig, CLIPConfig, BertModel
from transformers.models.clip import CLIPProcessor
from models import *
from processor import *
from trainer import *

sys.path.append("..")
warnings.filterwarnings("ignore", category=UserWarning)
#定义GPU环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
DATA_PROCESS_CLASS = {
    'bert-vit-inter-re': (MREProcessor, MREDataset),
    'bert': (MREProcessor, MREDataset),
    'vit': (MREProcessor, MREDataset),
}

MODEL_CLASS = {
    'HvFormer': BertVitInterReModel,
    'bert': BertForMultiLabelEntityClassification,
}

DATA_PATH = {
    'm2re': {'train': 'train.json',
             'dev': 'val.json',
             'test': 'test.json',

             },
}

IMG_PATH = {
    'm2re': {'train': '../mmdataset/dataset',
             'dev': '../mmdataset/dataset',
             'test': '../mmdataset/dataset'},
}
AUX_PATH = {
    'mnre': {
        'train': None,
        'dev': None,
        'test': None
    },
}

def set_seed(seed=2022):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def save_label_ids(label_ids, path):
    ids = label_ids.copy()
    ids.sort()
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "label.txt"), "w") as file:
        for id in ids:
            file.write(str(id) + "\n")
def load_previous_labeled_ids(current_round, last_round_logging_path):
    # read labeled ids
    if current_round == 0:
        return []
    else:
        with open(last_round_logging_path / "label.txt", "r") as file:
            lines = file.readlines()
            labeled_ids = [str(line).rstrip() for line in lines]
        return labeled_ids




def get_logger(args):
    if args.do_test:
        args.experiment_name = args.load_path
        args.load_path = os.path.join(args.save_path, args.experiment_name)
        log_filename = 'logs/'+ args.experiment_name
        if args.write_path is not None:
            args.write_path = os.path.join(args.write_path, args.experiment_name)
    else:
        args.experiment_name = args.experiment_name + "_" + args.dataset_name + '_' + time.strftime('%Y_%m_%d_%H_%M_%S')
        log_filename = "logs/" + args.experiment_name
        args.save_path = os.path.join(args.save_path, args.experiment_name)
        if args.write_path is not None:
            args.write_path = os.path.join(args.write_path, args.experiment_name)
    print(args)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=log_filename,
                        level=logging.INFO)
    return logging.getLogger(__name__)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default='test', type=str, help="The name of current experiment.")
    parser.add_argument('--model_name', default='HvFormer', type=str, help="The name of bert.")
    parser.add_argument('--vit_name', default='../dataset/clip-vit-base-patch32', type=str, help="The name of vit.")
    parser.add_argument('--dataset_name', default='m2re', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='../dataset/bert-base-uncased', type=str,
                        help="Pretrained language model name, bart-base or bart-large")
    parser.add_argument('--num_epochs', default=20, type=int, help="Training epochs")
    parser.add_argument('--device', default='cuda:0', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=128, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.06, type=float)
    parser.add_argument('--eval_begin_epoch', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--load_path', default='best_model', type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default="ckpt", type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str,
                        help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--do_train', action='store_false')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--prompt_len', default=4, type=int)
    parser.add_argument('--max_seq', default=80, type=int)
    parser.add_argument('--aux_size', default=128, type=int, help="aux size")
    parser.add_argument('--rcnn_size', default=64, type=int, help="rcnn size")

    parser.add_argument('--log_mode', dest='log_mode', default='logger', help='The way of printing logs produced in '
                                                                              'training and testing procedure')
    parser.add_argument('--num_workers', default=10, type=int, help="number of process workers for dataloader")

    parser.add_argument('--ignore_idx', default=0, type=int)
    parser.add_argument('--crf_lr', default=5e-2, type=float, help="learning rate")
    parser.add_argument('--prompt_lr', default=3e-4, type=float, help="learning rate")
    parser.add_argument('--current_round', default=0, type=int, help="current round of AL training")
    parser.add_argument('--initial_query_budget', default=10000, type=int, help="initial query budget")
    parser.add_argument('--query_budget', default=1000, type=int, help="query budget")
    parser.add_argument('--logging_root_dir', default='logs', type=str, help="root dir for logging")

    return parser.parse_args()

def init_and_train_bert_vit_re(args, logger):
    data_process, dataset_class = DATA_PROCESS_CLASS[args.model_name]
    model_class = MODEL_CLASS[args.model_name]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    if args.do_train:
        re_data_path, img_path, aux_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[
            args.dataset_name]
        re_path = './rel2id.json'

        #re_path = './data/dataset/jmere_rel2id.json'
        clip_vit, clip_processor, aux_processor, rcnn_processor = None, None, None, None
        clip_processor = CLIPProcessor.from_pretrained(args.vit_name)  # args.vit_name="openai/clip-vit-base-patch32"
        aux_processor = CLIPProcessor.from_pretrained(args.vit_name)  # args.vit_name="openai/clip-vit-base-patch32"
        aux_processor.feature_extractor.size, aux_processor.feature_extractor.crop_size = args.aux_size, args.aux_size  # aux_size=128
        rcnn_processor = CLIPProcessor.from_pretrained(args.vit_name)
        rcnn_processor.feature_extractor.size, rcnn_processor.feature_extractor.crop_size = args.rcnn_size, args.rcnn_size  # rcnn_size=64
        clip_model = CLIPModel.from_pretrained(args.vit_name)
        clip_vit = clip_model.vision_model
        processor = data_process(re_data_path, re_path, args.bert_name, clip_processor=clip_processor,
                                 aux_processor=aux_processor, rcnn_processor=rcnn_processor)


        train_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, aux_size=args.aux_size,
                                      rcnn_size=args.rcnn_size, mode='train',)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
        
        dev_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, aux_size=args.aux_size,
                                    rcnn_size=args.rcnn_size, mode='dev')
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
        
        test_dataset = dataset_class(processor, transform, img_path, aux_path, args.max_seq, aux_size=args.aux_size,
                                     rcnn_size=args.rcnn_size, mode='test', write_path=args.write_path,
                                     do_test=args.do_test)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=True)

        

        re_dict = processor.get_relation_dict()
        tokenizer = processor.tokenizer  # 30526 tokens, be used to convert textual tokens to ids
        # start training, validating and testing
        vision_config = CLIPConfig.from_pretrained(args.vit_name).vision_config
        text_config = BertConfig.from_pretrained(args.bert_name)
        bert = BertModel.from_pretrained(args.bert_name)
        clip_model_dict = clip_vit.state_dict()
        text_model_dict = bert.state_dict()

        model = model_class(re_label_mapping=re_dict,
                            tokenizer=tokenizer,
                            args=args,
                            vision_config=vision_config,
                            text_config=text_config,
                            clip_model_dict=clip_model_dict,
                            bert_model_dict=text_model_dict, )

        trainer = BertVitReTrainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader,
                                   re_dict=re_dict, model=model, args=args, logger=logger, writer=None)
        trainer.train()


if __name__ == "__main__":
    args = parse_argument()
    # set_seed(args.seed)  # set seed, default is 1
    logger = get_logger(args)
    logger.info(args)
    try:
        TRAINER = {
            'HvFormer': init_and_train_bert_vit_re,
        }
        if args.model_name in TRAINER.keys():
            TRAINER[args.model_name](args, logger)
        else:
            raise Exception(f"The model {args.model_name} is not implemented!")
    except Exception:
        traceback.print_exc()
        logger.info(traceback.print_exc())
