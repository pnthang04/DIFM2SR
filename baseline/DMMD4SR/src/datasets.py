
import random
import torch
import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix

from torch.utils.data import Dataset
from baseline.DMMD4SR.src.utils import neg_sample,get_user_seqs
import copy


def _read_recbole_inter(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fr:
        for line_no, line in enumerate(fr):
            line = line.strip()
            if not line:
                continue
            if line_no == 0 and ":" in line and "item_id_list" in line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Invalid RecBole inter row in {path}: {line}")
            user_token, seq_tokens, target_token = parts
            seq = seq_tokens.split() if seq_tokens else []
            rows.append((user_token, seq, target_token))
    return rows


def _feature_item_tokens(text_embedding_path, image_embedding_path):
    feature_path = text_embedding_path or image_embedding_path
    if not feature_path:
        return []
    if feature_path.endswith(".npy") or not feature_path.endswith(".pt"):
        features = np.load(feature_path, mmap_mode="r")
        return [str(i) for i in range(features.shape[0])]
    features = torch.load(feature_path, map_location="cpu")
    return [str(i) for i in range(int(features.shape[0]))]


def load_recbole_sgp_data(args):
    prefix = os.path.join(args.data_dir, args.data_name)
    train_rows = _read_recbole_inter(prefix + ".train.inter")
    valid_rows = _read_recbole_inter(prefix + ".valid.inter")
    test_rows = _read_recbole_inter(prefix + ".test.inter")

    item_tokens = set(_feature_item_tokens(args.text_embedding_path, args.image_embedding_path))
    for rows in (train_rows, valid_rows, test_rows):
        for _, seq, target in rows:
            item_tokens.update(seq)
            item_tokens.add(target)

    def sort_key(token):
        try:
            return (0, int(token))
        except ValueError:
            return (1, token)

    sorted_tokens = sorted(item_tokens, key=sort_key)
    token_to_id = {token: idx + 1 for idx, token in enumerate(sorted_tokens)}
    id_to_token = [None] + sorted_tokens

    def convert(rows):
        examples = []
        user_tokens = []
        for user_token, seq, target in rows:
            mapped_seq = [token_to_id[token] for token in seq]
            mapped_target = token_to_id[target]
            examples.append((mapped_seq, mapped_target))
            user_tokens.append(user_token)
        return examples, user_tokens

    train_examples, _ = convert(train_rows)
    valid_examples, valid_user_tokens = convert(valid_rows)
    test_examples, test_user_tokens = convert(test_rows)
    item_size = len(id_to_token) + 1

    def rating_matrix(examples):
        row, col, data = [], [], []
        for user_idx, (seq, _) in enumerate(examples):
            for item in seq:
                row.append(user_idx)
                col.append(item)
                data.append(1)
        return csr_matrix(
            (np.array(data), (np.array(row), np.array(col))),
            shape=(len(examples), item_size),
        )

    args.item_token_ids = id_to_token
    args.item_size = item_size
    args.mask_id = len(id_to_token)
    return {
        "train_examples": train_examples,
        "valid_examples": valid_examples,
        "test_examples": test_examples,
        "valid_user_tokens": valid_user_tokens,
        "test_user_tokens": test_user_tokens,
        "valid_rating_matrix": rating_matrix(valid_examples),
        "test_rating_matrix": rating_matrix(test_examples),
    }


class RecBoleSequentialDataset(Dataset):
    def __init__(self, args, examples, data_type="train"):
        self.args = args
        self.examples = examples
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _pad(self, values):
        values = values[-self.max_len:]
        return [0] * (self.max_len - len(values)) + values

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 1)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 1)
                inserted_sequence.append(item_id)
            inserted_sequence.append(item)
        return inserted_sequence

    def __getitem__(self, index):
        seq, target = self.examples[index]
        if self.data_type == "test":
            full_seq = self._add_noise_interactions(seq + [target])
            input_ids = full_seq[:-1]
            answer = full_seq[-1]
        else:
            input_ids = seq
            answer = target

        target_pos = input_ids[1:] + [answer]
        input_ids = self._pad(input_ids)
        target_pos = self._pad(target_pos)
        answer_tensor = [answer]

        if self.data_type == "train":
            return (
                torch.tensor(index, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(answer_tensor, dtype=torch.long),
            )

        return (
            torch.tensor(index, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(answer_tensor, dtype=torch.long),
        )

    def __len__(self):
        return len(self.examples)


class Generate_tag():
    def __init__(self,data_path,data_name,save_path):
        self.path=data_path
        self.data_name=data_name+"_1"
        self.save_path=save_path
    def generate(self):
        data_f=self.path+"/"+self.data_name+".txt"
        train_dic={}
        valid_dic={}
        test_dic={}
        with open(data_f,"r") as fr:
            data=fr.readlines()
            for d_ in data:
                items=d_.split(' ')
                tag_train=int(items[-3])
                tag_valid=int(items[-2])
                tag_test=int(items[-1])
                train_temp=list(map(int, items[:-3]))
                valid_temp=list(map(int, items[:-2]))
                test_temp=list(map(int,items[:-1]))
                if tag_train not in train_dic:
                    train_dic.setdefault(tag_train,[])
                train_dic[tag_train].append(train_temp)
                if tag_valid not in valid_dic:
                    valid_dic.setdefault(tag_valid,[])
                valid_dic[tag_valid].append(valid_temp)
                if tag_test not in test_dic:
                    test_dic.setdefault(tag_test,[])
                test_dic[tag_test].append(test_temp)

        total_dic={"train":train_dic,"valid":valid_dic,"test":test_dic}
        print("Saving data to ", self.save_path)
        with open(self.save_path+"/"+self.data_name+"_t.pkl","wb") as fw:
            pickle.dump(total_dic, fw)

    def load_dict(self,data_path):
        if not data_path:
            raise ValueError('invalid path')
        elif not os.path.exists(data_path):
            print("The dict not exist, generating...")
            self.generate()
        with open(data_path, 'rb') as read_file:
            data_dict = pickle.load(read_file)
        return data_dict

    def get_data(self,data_path,mode):
        data=self.load_dict(data_path)
        return data[mode]




class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train", similarity_model_type="offline"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

        # create target item sets
        self.sem_tag=Generate_tag(self.args.data_dir,self.args.data_name,self.args.data_dir)
        self.train_tag=self.sem_tag.get_data(self.args.data_dir+"/"+self.args.data_name+"_1_t.pkl","train")
        self.true_user_id,_,_,_,_=get_user_seqs(args.train_data_file)

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids =[0] * pad_len+copied_input_ids
        copied_input_ids=copied_input_ids[-self.max_len:]

        if type(target_pos)==tuple:
            pad_len_1=self.max_len-len(target_pos[1])
            target_pos_1 =   [0] * pad_len+target_pos[0]
            target_pos_2=   [0] * pad_len_1+target_pos[1]
            target_pos_1 = target_pos_1[-self.max_len:]
            target_pos_2 = target_pos_2[-self.max_len:]
            assert len(target_pos_1) == self.max_len
            assert len(target_pos_2) == self.max_len
        else:
            target_pos =  [0] * pad_len+target_pos
            target_pos = target_pos[-self.max_len:]
            assert len(target_pos) == self.max_len

        assert len(copied_input_ids) == self.max_len
        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            if type(target_pos)==tuple:
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos_1, dtype=torch.long),
                    torch.tensor(target_pos_2, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )

            else:
                cur_rec_tensors = (
                    torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                    torch.tensor(copied_input_ids, dtype=torch.long),
                    torch.tensor(target_pos, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                )
        return cur_rec_tensors


    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        t_user_id=self.true_user_id[index]
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            temp=self.train_tag[items[-3]]
            flag=False
            for t_ in temp:
                if t_[1:]==items[:-3]:
                    continue
                else:
                    target_pos_=t_[1:]
                    flag=True
            if not flag:
                target_pos_=random.choice(temp)[1:]
            seq_label_signal = items[-2] # no use
            answer = [0]  # no use
        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            items_with_noise = self._add_noise_interactions(items)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]
        if self.data_type == "train":
            target_pos=(target_pos,target_pos_)
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            return (cur_rec_tensors)
        elif self.data_type == "valid":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items_with_noise, input_ids, target_pos, answer)
            return cur_rec_tensors

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.user_seq)




# Dynamic Segmentation operations
def DS_default(i_file,o_file):
    """
    :param i_file: original data
    :param o_file: output data
    :return:
    """
    with open(i_file,"r+") as fr:
        data=fr.readlines()
    aug_d={}
    for d_ in data:
        u_i,item=d_.split(' ',1)
        item=item.split(' ')
        item[-1]=str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start=0
        j=3
        if len(item)>53:
            while start<len(item)-52:
                j=start+4
                while j<len(item):
                    if start<1 and j-start<53:
                        aug_d[u_i].append(item[start:j])
                        j+=1
                    else:
                        aug_d[u_i].append(item[start:start+53])
                        break
                start+=1
        else:
            while j<len(item):
                aug_d[u_i].append(item[start:j+1])
                j+=1
    with open(o_file,"w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i+" "+' '.join(i_)+"\n")



# Dynamic Segmentation operations
def DS(i_file,o_file,max_len):
    """
    :param i_file: original data
    :param o_file: output data
    :max_len: the max length of the sequence
    :return:
    """
    with open(i_file,"r+") as fr:
        data=fr.readlines()
    aug_d={}
    # training, validation, and testing
    max_save_len=max_len+3
    # save
    max_keep_len=max_len+2
    for d_ in data:
        u_i,item=d_.split(' ',1)
        item=item.split(' ')
        item[-1]=str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start=0
        j=3
        if len(item)>max_save_len:
            # training, validation, and testing
            while start<len(item)-max_keep_len:
                j=start+4
                while j<len(item):
                    if start<1 and j-start<max_save_len:
                        aug_d[u_i].append(item[start:j])
                        j+=1
                    else:
                        aug_d[u_i].append(item[start:start+max_save_len])
                        break
                start+=1
        else:
            while j<len(item):
                aug_d[u_i].append(item[start:j+1])
                j+=1
    with open(o_file,"w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i+" "+' '.join(i_)+"\n")


class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)




if __name__ == "__main__":
    # dynamic segmentation
    DS("../data/Beauty.txt","../data/Beauty_1.txt",10)
    # DS_default("../data/Beauty.txt", "../data/Beauty_1.txt")
    # generate target item
    g=Generate_tag("../data","Beauty","../data")
    # generate the dictionary
    data=g.get_data("../data/Beauty_1_t.pkl","train")
    i=0
    # Only one sequence in the data dictionary in the training phase has the target item ID
    for d_ in data:
        if len(data[d_])<2:
            i+=1
            print("less is : ",data[d_],d_)
    print(i)



