import os
import numpy as np
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import (
    RecWithContrastiveLearningDataset,
    RecBoleSequentialDataset,
    DS,
    load_recbole_sgp_data,
)

from baseline.DMMD4SR.src.trainers import ICLRecTrainer
from baseline.DMMD4SR.src.models import SASRecModel
from baseline.DMMD4SR.src.utils import EarlyStopping, get_user_seqs, check_path, set_seed


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        value = getattr(args, arg)
        if isinstance(value, (list, tuple, dict, set)):
            value = f"{type(value).__name__}(len={len(value)})"
        print(f"{arg:<30} : {str(value):>35}")


def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--data_dir", default="../process_data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Toys", type=str)  # [Sports_and_Outdoors, Beauty, Toys_and_Games, Home]
    parser.add_argument("--data_format", default="sequence", choices=["sequence", "recbole"], type=str)
    parser.add_argument("--encoder",default="SAS",type=str) 
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--model_idx", default=0, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # robustness experiments
    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="percentage of negative interactions in a sequence - robustness analysis",
    )

    ## contrastive learning task args
    parser.add_argument(
        "--temperature", default=1.0, type=float, help="softmax temperature (default:  1.0) - not studied."
    )


    parser.add_argument(
        "--sim",default='dot',type=str,help="the calculate ways of the similarity."
    )

    # model args
    parser.add_argument("--model_name", default="ICLRec", type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--n_clusters", type=int, default=32, help="number of clusters")
    parser.add_argument("--lambda_history", type=float, default=0.6, help="lambda of history")
    parser.add_argument("--lambda_intent", type=float, default=0.4, help="lambda of intent")


    # encoder
    parser.add_argument("--num_experts", type=int, default=2, help="number of num_experts")
    parser.add_argument("--diff_loss", type=float, default=0.01)
    parser.add_argument("--icl_loss", type=float, default=0.001)

    # multi modal
    parser.add_argument("--is_use_mm", type=bool, default=True, help="is use mm embedding")
    parser.add_argument("--is_use_text", type=bool, default=True, help="is use text embedding")
    parser.add_argument("--is_use_image", type=bool, default=False, help="is use image embedding")
    parser.add_argument("--text_embedding_path", default='../process_data/text_features_beauty.pt', type=str)
    parser.add_argument("--image_embedding_path", default='../process_data/image_features_beauty.pt', type=str)
    parser.add_argument("--pretrain_emb_dim", type=int, default=512, help="pretrain_emb_dim of clip model")
    parser.add_argument("--is_use_cross", type=bool, default=True, help="is use mm cross")

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=80, help="number of epochs")
    parser.add_argument("--early_stop_num", type=int, default=10, help="number of early_stop")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2024, type=int)
    # loss weight
    parser.add_argument("--rec_weight", type=float, default=1.0, help="weight of contrastive learning task")

    # ablation experiments
    parser.add_argument("--cl_mode",type=str,default='cf',help="contrastive mode")
    # {'cf':coarse-grain+fine-grain,'c':only coarse-grain,'f':only fine-grain}
    parser.add_argument("--f_neg", action="store_true", help="delete the FNM component (both in cicl and ficl)")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")

    args = parser.parse_args()
    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    if args.data_format == "recbole":
        recbole_data = load_recbole_sgp_data(args)
        if args.pretrain_emb_dim == 512:
            if args.text_embedding_path.endswith(".npy") or not args.text_embedding_path.endswith(".pt"):
                args.pretrain_emb_dim = int(np.load(args.text_embedding_path, mmap_mode="r").shape[1])
            else:
                args.pretrain_emb_dim = int(torch.load(args.text_embedding_path, map_location="cpu").shape[1])
    else:
        args.data_file = args.data_dir + args.data_name + ".txt"
        args.train_data_file = args.data_dir + args.data_name + "_1.txt"

        # construct supervisory signals via DS(·) operation
        if not os.path.exists(args.train_data_file):
            DS(args.data_file,args.train_data_file,args.max_seq_length)

        # training data
        _,train_user_seq, _, _, _ = get_user_seqs(args.train_data_file)
        # valid and test data
        _,user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

        args.item_size = max_item + 2
        args.mask_id = max_item + 1
    # save model args
    args_str = f"{args.model_name}-{args.encoder}-{args.data_name}-{args.model_idx}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")

    show_args_info(args)

    with open(args.log_file, "a") as f:
        f.write(str(args) + "\n")

    # set item score in train set to `0` in validation
    if args.data_format == "recbole":
        args.train_matrix = recbole_data["valid_rating_matrix"]
    else:
        args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # cluster
    if args.data_format == "recbole":
        cluster_dataset = RecBoleSequentialDataset(
            args, recbole_data["train_examples"], data_type="train"
        )
    else:
        cluster_dataset = RecWithContrastiveLearningDataset(
            args, train_user_seq, data_type="train"
        )
    cluster_sampler = SequentialSampler(cluster_dataset)
    cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size)

    # training data
    if args.data_format == "recbole":
        train_dataset = RecBoleSequentialDataset(
            args, recbole_data["train_examples"], data_type="train"
        )
    else:
        train_dataset = RecWithContrastiveLearningDataset(
            args, train_user_seq, data_type="train"
        )
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    if args.data_format == "recbole":
        eval_dataset = RecBoleSequentialDataset(
            args, recbole_data["valid_examples"], data_type="valid"
        )
    else:
        eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    if args.data_format == "recbole":
        test_dataset = RecBoleSequentialDataset(
            args, recbole_data["test_examples"], data_type="test"
        )
    else:
        test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)


    model = SASRecModel(args=args)
    trainer = ICLRecTrainer(model, train_dataloader,cluster_dataloader, eval_dataloader, test_dataloader, args)

    if args.do_eval:
        if args.data_format == "recbole":
            trainer.args.train_matrix = recbole_data["test_rating_matrix"]
        else:
            trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)

        print(f"Load model from {args.checkpoint_path} for test!")
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        print(f"Train ICLRec")
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.early_stop_num, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        if args.data_format == "recbole":
            trainer.args.train_matrix = recbole_data["test_rating_matrix"]
        else:
            trainer.args.train_matrix = test_rating_matrix
        print("---------------Change to test_rating_matrix!-------------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, "a") as f:
        f.write(args_str + "\n")
        f.write(result_info + "\n")


main()
