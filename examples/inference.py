import torch.nn as nn
import argparse
import logging
import pickle
from multiprocessing import Pool
from matplotlib.colors import LinearSegmentedColormap

from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset,  SequentialSampler
from tqdm import tqdm
import os
import torch
from torch.nn.functional import softmax
import random
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import pandas as pd
logger = logging.getLogger(__name__)
import csv
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DNATokenizer,
    PreTrainedTokenizer,
    BertForGenerate,
    BertForMaskedLM_ori

)


MODEL_CLASSES = {
    "dnabert_ori": (BertConfig, BertForMaskedLM_ori, DNATokenizer),
    "dna_encoder": (BertConfig, BertForMaskedLM, DNATokenizer),
    "dna_generate": (BertConfig, BertForGenerate, DNATokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
}




def convert_line_to_example(tokenizer, lines, max_length, add_special_tokens=True):
    examples = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens, max_length=max_length)[
        "input_ids"]
    return examples


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer:PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", file_path)

            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

            if args.n_process == 1:
                self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)[
                    "input_ids"]
            else:
                n_proc = args.n_process
                p = Pool(n_proc)
                indexes = [0]
                len_slice = int(len(lines) / n_proc)
                for i in range(1, n_proc + 1):
                    if i != n_proc:
                        indexes.append(len_slice * (i))
                    else:
                        indexes.append(len(lines))
                results = []
                for i in range(n_proc):
                    results.append(p.apply_async(convert_line_to_example,
                                                 [tokenizer, lines[indexes[i]:indexes[i + 1]], block_size, ]))
                    print(str(i) + " start")
                p.close()
                p.join()

                self.examples = []
                for result in results:
                    ids = result.get()
                    self.examples.extend(ids)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)





def t_SNE_sklearn(all_seq_embedding_list=None, all_seq_embedding_path=None, save_data=None):
    """
    将所有序列的embedding用T-SNE降维并且可视化，使用sklearn降维只适用于少量数据而不是用于大规模数据
     @param all_seq_embedding_path:
     @param all_seq_embedding_list:
    :param all_seq_embedding_path:所有序列的embedding路径
    :return:无
    """

    if save_data and os.path.exists(save_data):
        with open(save_data, "rb") as f:
            embedded_data = pickle.load(f)
    else:
        if all_seq_embedding_path:
            with open(all_seq_embedding_path, "rb") as f:
                seq_embeddings = pickle.load(f)
        elif all_seq_embedding_list:
            seq_embeddings = all_seq_embedding_list
        else:
            logger.info("have no seq embedding informations")

        data = []
        for seq in seq_embeddings:
            if isinstance(seq, list):
                data.append(seq)
            else:
                data.append(seq.cpu().numpy().flatten())
        data = np.array(data)
        # 使用 t-SNE 进行降维，降到 2 维
        tsne = TSNE()
        embedded_data = tsne.fit_transform(data)



        if save_data:
            with open(save_data, "wb") as f:
                pickle.dump(embedded_data, f)
    return embedded_data



def HC_HEBO(seq_KD, seq_embedding=None, embedding_cache_dir=None,max=True,num_to_gen=100):
    logger.info("Using HEBO.csv to optimize our gneration!\n")

    # 处理embedding cache
    if embedding_cache_dir:
        with open(embedding_cache_dir, "rb") as f:
            embeds = pickle.load(f)
    else:
        embeds = seq_embedding
    X_init = []
    Y_init = []  # 模拟观测的目标值
    seqs_X_Y = {}

    for seq in seq_KD.keys():
        X_init.append(embeds[seq].numpy().flatten())
        if max:
            Y_init.append(-float(seq_KD[seq]))
        else:
            Y_init.append(float(seq_KD[seq]))
        seqs_X_Y[seq] = list([embeds[seq].numpy(), float(seq_KD[seq])])

    hidden_size = embeds[seq].size()[0] * embeds[seq].size()[1]
    columns_list = ["x" + str(i) for i in range(hidden_size)]
    X_init = pd.DataFrame(np.array(X_init), columns=columns_list)
    Y_init = np.array(Y_init)
    Y_init = Y_init.reshape(-1, 1)
    params = [{'name': "x" + str(i), 'type': 'num', 'lb': np.min(X_init-5, axis=0)[i], 'ub': np.max(X_init+5, axis=0)[i]} for i in range(hidden_size)]

    space = DesignSpace().parse(params)

    opt = HEBO(space, model_name="gp")
    opt.observe(X_init, Y_init)

    rec = opt.suggest(n_suggestions=num_to_gen)
    candidates = np.array(rec)
    candidates_list = []
    for i in range(candidates.shape[0]):
        candidates_list.append(candidates[i, :])

    print("HEBO has been done!")
    # 返回选定点及其值
    return candidates_list



def seq_embedding_Kd(seq_file_dir, embedding_dict, ):


    def kmers_sliding_windows(kmers, seq):
        return " ".join([seq[i:i + kmers] for i in (range(len(seq) - kmers))])

    seq_embedding = {}
    with open(seq_file_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            seq, Kd = line.strip().split(sep=",")
            seq_embedding[seq] = embedding_dict[seq]

    return seq_embedding





from torch_clustering import  PyTorchGaussianMixture


def GMM_embedding_clustering_pytorch(sequence_embedding, center_save_path, k=10,seed=42):
    data = []
    
    for seq in sequence_embedding:
        data.append(seq.cpu().numpy().flatten())

    X = torch.tensor(data).cuda()

    gmm = PyTorchGaussianMixture(n_clusters=k)
    pre_label = gmm.fit_predict(X)
    pre_label = torch.argmax(torch.tensor(pre_label), dim=1)
    centers = gmm.cluster_centers_
    pre_label = pre_label.detach().cpu().numpy()
    centers = centers.detach().cpu().numpy()

    return centers, pre_label



def kmer2seq(kmers: int, seq: list):
    for i in range(len(seq)):
        if i == 0:
            real_seq = seq[0]
        else:
            real_seq += seq[i][2]
    return real_seq


def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(seq))

def DNA_socre_com(seq, score_file_dir_or_sequecne_list,motif_len=8,):
    motif_score = {}
    if os.path.isfile(score_file_dir_or_sequecne_list):
        with open(score_file_dir_or_sequecne_list, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "ID" in line:
                    continue
                line_list = line.strip().split()
                motif_score[line_list[0]] = line_list[-1]
                motif_score[reverse_complement(line_list[0])] = line_list[-1]

    motif_list = []
    total_score = 0

    for idx in range(0, len(seq) - motif_len + 1):
        motif = seq[idx:idx + motif_len]

        if motif in motif_score:
            motif_list.append(motif)
            total_score += float(motif_score[motif])
    if len(motif_list) == 0:
        print(f"{seq}have no sub seq")
        return 0
    
    print(str(seq) + "," + str(total_score))
    return total_score




def embedding_generation():
    parser = argparse.ArgumentParser()

    # Required parameter
    parser.add_argument(
        "--data_dir", default="datasets/CELL_SELEX_2013/Srebf1_TACATT20NCG_Z_4/all_seq_3mers.txt", type=str,
        help="The input training data file (a text file),the format is DNA sequecne without kmers seperated."
    )

    parser.add_argument(
        "--model_name_or_path",
        default="model_save_path/Srebf1/encoder/checkpoint",
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--decode_model_name_or_path",
        default="model_save_path/Srebf1/decoder/checkpoint",
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )





    parser.add_argument(
        "--model_type", default="dna_encoder", type=str, help="The model architecture to be trained or fine-tuned.",
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )

    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )


    parser.add_argument("--n_process", type=int, default=8, help="")

    parser.add_argument("--batch_size", type=int, default=32, help="")

    parser.add_argument("--visulization_tSNE", type=bool, default=False,
                        help="whether to use t-SNE to visulize the embedding.")

    parser.add_argument("--Kd_seq_path", type=str, default="datasets/CELL_SELEX_2013/Srebf1_TACATT20NCG_Z_4/HC-HEBO_data/seq-func.csv",
                        help="the sequnce with kd file path ,the file is .csv file and formate is \"ACTGGACTGGACTGGACTGG,12.3 \"")

    parser.add_argument(
        "--Kd_seq_data_dir", default="datasets/CELL_SELEX_2013/Srebf1_TACATT20NCG_Z_4/HC-HEBO_data/seq_3mer.csv", type=str,
        help="The input training data file (a text file),the format is DNA sequecne with kmers seperated."
    )

    parser.add_argument("--embedding_len", type=int, default=20, help="the embedding lenth ")

    parser.add_argument("--seq_len", type=int, default=20, help="the lenth of the sequence to generate")

    parser.add_argument("--bo_cycles", type=int, default=100, help="the number of the sequence to generate")

    parser.add_argument("--BO_linebyline", type=bool, default=True, help="BO line by line")


    parser.add_argument("--density_sample", type=str, default=False, help="density sampleto gen")

    parser.add_argument("--select_gmm_selex_dir", type=str, default=False, help="select gmm&selex")

    parser.add_argument("--GMM_Cluster", type=str, default=False,
                        help="Kmeans center")

    parser.add_argument("--mask_gen", type=bool, default=True, help="Mask model to generate")

    parser.add_argument("--use_HEBO", type=bool, default=True, help="HEBO.csv to optimize the DNA genneration or not")

    parser.add_argument("--model_down_dim", type=int, default=8,
                        help="the fintuned model embedding dim ")

    parser.add_argument("--label_dir", type=str, default="datasets/CELL_SELEX_2013/Srebf1_TACATT20NCG_Z_4/PBM_label_data/pTH0914_HK_8mer.raw",help="the PBM labels data dir")

    parser.add_argument("--f_linker", type=str,
                        default="TACATT",
                        help="the forward linker")

    parser.add_argument("--r_linker", type=str,
                        default="CG",
                        help="the reverse linker")

    
    args = parser.parse_args()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.model_name_or_path:
        encoder_model = model_class.from_pretrained(
            args.model_name_or_path,
            down_dim=args.model_down_dim,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config

        )
    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_model.to(device)

    next_point_list = []

    with torch.no_grad():
        sequence_embedding = {}
        if args.data_dir:
            # if os.path.exists("all_seq_embedding_cache"):
            #     with open("all_seq_embedding_cache","rb")as f:
            #         sequence_embedding=pickle.load(f)
            # else:
            input_datasets = LineByLineTextDataset(tokenizer, args, file_path=args.data_dir)
            # input_datasets =TIM_dataset(args.data_dir,tokenizer)
            sampler = SequentialSampler(input_datasets)
            dataloader = DataLoader(input_datasets, sampler=sampler, batch_size=args.batch_size)

            tmp_total_embedding_tensor = None

            all_batch_sequence_dict = {}
            for batch in tqdm(dataloader, total=len(dataloader), desc="model inference"):

                input = batch[:, :].to(device)
                output = encoder_model(input)
                hidden_state = output[1]

                if tmp_total_embedding_tensor == None:
                    tmp_total_embedding_tensor = hidden_state.view(hidden_state.shape[0], -1)
                else:
                    tmp_total_embedding_tensor = torch.cat(
                        [tmp_total_embedding_tensor, hidden_state.view(hidden_state.shape[0], -1)], dim=0)

                for idx in range(input.size()[0]):
            
                    decoded_text = kmer2seq(3, tokenizer.convert_ids_to_tokens(input[:, 1:,][idx]))
                    all_batch_sequence_dict[decoded_text] = hidden_state[idx, :].to("cpu")

                # sequence_embedding
            sequence_embedding = [tmp_total_embedding_tensor[batch_idx:batch_idx + 1] for batch_idx in
                                  range(len(tmp_total_embedding_tensor))]
          
            if args.GMM_Cluster:
              
                next_point_list, Clustering_label = GMM_embedding_clustering_pytorch(sequence_embedding,
                                                                                     args.GMM_Cluster)
                all_next_point_list = list(next_point_list)

                if args.select_gmm_selex_dir:
                    act_labels=[]
                    seqs=[]
                    with open(args.select_gmm_selex_dir,"r")as f:
                        lines= f.readlines()
                        for line in lines:
                            seq,act=line.strip().split(sep=",")
                            seqs.append(seq)
                            act_labels.append(float(act))
                    data = {
                        'DNA_sequence': seqs,
                        'Cluster_label': Clustering_label,
                        'Activity_label': act_labels
                    }
                    df = pd.DataFrame(data)
                    result = df.loc[df.groupby('Cluster_label')['Activity_label'].idxmax()]
                    with open(os.path.dirname(args.select_gmm_selex_dir)+"/selex_output.csv","w")as f:
                        for index, row in result.iterrows():
                            print(
                                f"Cluster {row['Cluster_label']}: DNA Sequence - {row['DNA_sequence']} (Activity: {row['Activity_label']})")
                            f.write(f"{row['DNA_sequence']},{row['Activity_label']}\n")
                    return
            from sklearn.neighbors import KernelDensity
            def embedding_sampling_with_density(embeddings, size):
                """
                根据嵌入的概率密度进行采样

                参数：
                    embeddings: numpy.array，表示嵌入的数组，每行是一个嵌入向量
                    size: int，采样的数量

                返回值：
                    samples: numpy.array，采样得到的嵌入数组
                """
                if torch.is_tensor(embeddings):
                    embeddings=embeddings.cpu().numpy()



                # 使用 Kernel Density Estimation 估计嵌入的概率密度
                kde = KernelDensity(bandwidth=0.5).fit(embeddings)
                # 从估计的概率密度中采样
                sampled_embeddings = kde.sample(n_samples=size)
                return sampled_embeddings
            if args.density_sample:
                next_point_list=embedding_sampling_with_density(tmp_total_embedding_tensor,2000)
                all_next_point_list = next_point_list.tolist()
                all_next_point_list=[torch.tensor(item) for item in all_next_point_list]


      
        Kd_seq_batch_sequence_dict = {}
        if args.Kd_seq_data_dir:
            args.n_process = 1
            input_datasets = LineByLineTextDataset(tokenizer, args, file_path=args.Kd_seq_data_dir)
            # input_datasets =TIM_dataset(args.data_dir,tokenizer)
            sampler = SequentialSampler(input_datasets)
            dataloader = DataLoader(input_datasets, sampler=sampler, batch_size=1)

            tmp_total_embedding_tensor = None

            with torch.no_grad():
                for batch in tqdm(dataloader, total=len(dataloader), desc="model inference Kd sequence"):

                    input = batch[:, :].to(device)
                    output = encoder_model(input)
                    hidden_state = output[1]

                    if tmp_total_embedding_tensor == None:

                        tmp_total_embedding_tensor = hidden_state.view(hidden_state.shape[0], -1)
                    else:
                        tmp_total_embedding_tensor = torch.cat(
                            [tmp_total_embedding_tensor, hidden_state.view(hidden_state.shape[0], -1)], dim=0)

                    for idx in range(input.size()[0]):
                        # if idx==input.size()[0]:
                        #     print("sadasd")
                        #     break
                        decoded_text = kmer2seq(3, tokenizer.convert_ids_to_tokens(input[:, 1:-1][idx]))
                        Kd_seq_batch_sequence_dict[decoded_text] = hidden_state[idx, :, :].to("cpu")

                    # sequence_embedding
                kd_sequence_embedding = [tmp_total_embedding_tensor[batch_idx:batch_idx + 1] for batch_idx in
                                         range(len(tmp_total_embedding_tensor))]

        if args.visulization_tSNE:
            if args.data_fre_dir:
                with open(args.data_fre_dir, "r") as f:
                    text = f.readlines()
                real_labels = [int(float(line.strip())) for line in text]
                real_labels = np.array(real_labels)
            if args.GMM_Cluster:
                clustering_labels = np.array(Clustering_label)
            # results=compute_evaluate_clustering(sequence_embedding, real_labels, clustering_labels)
            # embeddings=[*sequence_embedding,*generated_sequence_embedding,*kd_sequence_embedding]
            embeddings = [*sequence_embedding]
            embedded_data = t_SNE_sklearn(embeddings)
            tSNE_result = np.concatenate((embedded_data, clustering_labels[:,np.newaxis]),axis=1)
            np.savetxt('tSNE_xy.csv', tSNE_result, delimiter=',')


            """using clustering label to release following code"""
            unique_labels = np.unique(Clustering_label)
            colors = plt.cm.rainbow(np.linspace(0 , 1 , len(unique_labels)))
            for label , color in zip(unique_labels , colors):
                if label == -1:
                    mask = (Clustering_label == label)
                    plt.scatter(embedded_data[mask , 0] , embedded_data[mask , 1] , color='black' , label=f'Noise (Cluster {label})')
                else:
                    mask = (Clustering_label == label)
                    plt.scatter(embedded_data[mask , 0] , embedded_data[mask , 1] , color=color , label=f'Cluster {label}')

            fig = plt.figure(figsize=(10, 8), dpi=300)
            colors_bluetored = [(0.6, 0.8, 1.0), (0, 0, 1), (1, 0, 0)]
            colors_redtoblue = [(1, 0, 0), (0, 0, 1), (0.6, 0.8, 1.0), ]
            n_bins = 2  # 可以根据需要调整 bin 数量
            cmap_name = "blue_to_red"
            cmap = LinearSegmentedColormap.from_list(cmap_name, colors_bluetored, N=n_bins)
            cmap = 'tab10'
            plt.xticks(ticks=range(int(min(embedded_data[:, 1])), int(max(embedded_data[:, 1])) + 1,50), fontsize=30)
            plt.yticks(ticks=range(int(min(embedded_data[:, 0])), int(max(embedded_data[:, 0])) + 1,50), fontsize=30)
            for spine in plt.gca().spines.values():
                spine.set_linewidth(2)
            # 绘制降维后的数据
            plt.axis('off')

            def map_size(value):
                if value ==0:
                    return 's'  # 小正方形
                elif value==1 :
                    return '^'  # 小三角形
                else:
                    return 'o'  # 小圆形


            plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=clustering_labels, cmap=cmap, s=100, edgecolor='white' )
            plt.show()



        
        """
        BO Part
        """

    if args.Kd_seq_path and args.Kd_seq_data_dir:  # do BO and GPR
        seq_KD = {}
        with open(args.Kd_seq_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                seq, kd = line.strip().split(sep=",")
                seq_KD[seq] = kd
        seq_embedding = seq_embedding_Kd(args.Kd_seq_path, Kd_seq_batch_sequence_dict, )
        all_next_point_list=[]
        if args.BO_linebyline:

            for seq,Kd in seq_KD.items():
                evaluate_bo_dict = {}
                evaluate_bo_dict[seq]=Kd
                print("="*50)
                logger.warning("Now the BO evaluate sequence is %s and its Kd is %s ",seq,Kd)

             
                next_point_list = HC_HEBO(evaluate_bo_dict, seq_embedding=seq_embedding,num_to_gen=args.bo_cycles)
                all_next_point_list.extend(next_point_list)

        else:
       
            next_point_list = HC_HEBO(seq_KD, seq_embedding=seq_embedding, num_to_gen=args.bo_cycles)
            all_next_point_list.extend(next_point_list)
           

    if args.decode_model_name_or_path:
        full_model = BertForGenerate.from_pretrained(
            args.decode_model_name_or_path,
            down_dim=args.model_down_dim,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config
        )

    encode_model = nn.Sequential(*list(full_model.children())[:3])
    decode_model = nn.Sequential(*list(full_model.children())[3:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_model.to(device)
    encode_model.to(device)
    decode_model.to(device)

    generated_seq_list = []
    if args.Kd_seq_path and args.Kd_seq_data_dir:  # do BO
        for next_location in tqdm(all_next_point_list,desc="sequence is generating!"):
            

            next_location = torch.tensor(next_location)
            next_location = next_location.to(torch.float32).to(device)
            next_location = next_location.reshape(1, args.embedding_len, args.model_down_dim)

            with torch.no_grad():
                # output=decode_model(next_location)
                # print(output)
                if args.mask_gen:
                    generated_seq = mask_seq_genernate_nolinkers(next_location, decode_model, full_model, encoder_model,
                                                       tokenizer,pre_link=args.f_linker,rever_link=args.r_linker)

                generated_seq_list.append(generated_seq)
    else:
        for next_location in tqdm(all_next_point_list,desc="sequence is generating!"):
            next_location = torch.tensor(next_location)
            next_location = next_location.to(torch.float32).to(device)
            next_location = next_location.reshape(1, args.embedding_len, args.model_down_dim)

            with torch.no_grad():
                # output=decode_model(next_location)
                # print(output)
                if args.mask_gen:
                    generated_seq = mask_seq_genernate_nolinkers(next_location, decode_model, full_model, encoder_model,
                                                       tokenizer,pre_link=args.f_linker,rever_link=args.r_linker)

                generated_seq_list.append(generated_seq)

  


    with open("BO_generated_sequences.csv", "w") as f:
        for idx,seq in enumerate(generated_seq_list):
            score = DNA_socre_com(seq,score_file_dir_or_sequecne_list=args.label_dir)
            if score != 0:
                f.write(f"{seq},{score}\n")




def top_k_top_p_sampling(logits, top_k=5, top_p=0.9, temperature=1.2):
    # 使用softmax计算概率分布
    probs = softmax(logits / temperature, dim=-1)

    # Top-K采样
    topK_value, topK_indices = torch.topk(probs, top_k)
    topK_value = softmax(topK_value, dim=-1)

    # 根据Top-P进行截断
    # sorted_probs , sorted_indices = torch.sort(topK_value , descending=True)
    cumulative_probs = torch.cumsum(topK_value, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # sorted_indices_to_remove[: , top_k:] = 1
    # sorted_indices[cumulative_probs > top_p] = -1
    sorted_indices_to_remove = ~sorted_indices_to_remove * topK_value
    sorted_indices_to_remove[sorted_indices_to_remove == 0] = float("-inf")
    # 随机选择一个词
    sampled_index = torch.multinomial(softmax(sorted_indices_to_remove.squeeze(), dim=-1), 1)
    # sampled_index=topK_indices[: , sampled_index]
    gen_seq = []
    for idx, i in enumerate(sampled_index.tolist()):
        gen_seq.append(topK_indices[:, idx, i])
    gen_seq = torch.unsqueeze(torch.tensor(gen_seq, device="cuda"), dim=0)
    return gen_seq


def mask_seq_genernate(input_embedding, decode_model, full_model, encode_model, tokenizer, lowest_probs_mask_ratio=0.15,
                       iterative_optimize=3, top_k=20, top_p=0.8, temperature=1, pre_link="TAAAAT", rever_link="AAA"):
    logits = decode_model(input_embedding)
    once_mask_token_probability = torch.softmax(logits, dim=2)
    # once_predict_seq_id = torch.argmax(once_mask_token_probability , dim=2)en
    once_predict_seq_id = top_k_top_p_sampling(logits, top_k, top_p, temperature)

    def kmers_sliding_windows(seq, kmers=3):
        return " ".join([seq[i:i + kmers] for i in (range(len(seq) - kmers + 1))])

    once_predict_seq_id[0, 0] = tokenizer.cls_token_id
    once_predict_seq_id[0, -1] = tokenizer.sep_token_id

    if pre_link and len(pre_link) > 3:
        pre_token_ids = tokenizer.batch_encode_plus(kmers_sliding_windows(pre_link).split())
        pre_link_len = len(pre_token_ids["input_ids"])
        for idx, token_id in enumerate(pre_token_ids["input_ids"]):
            once_predict_seq_id[0, idx + 1] = token_id[0]
    else:
        pre_link_len = 0

    if rever_link and len(rever_link) > 3:
        rever_token_ids = tokenizer.batch_encode_plus(kmers_sliding_windows(pre_link).split())
        rever_link_len = len(rever_token_ids["input_ids"])
        for idx, token_id in enumerate(rever_token_ids["input_ids"]):
            once_predict_seq_id[0, -1 - idx - rever_link_len] = token_id[0]
    else:
        rever_link_len = 0

    ##print first generated sequence
    decoded_tokens = tokenizer.convert_ids_to_tokens(once_predict_seq_id.view(-1))[1:-1]
    decoded_text = kmer2seq(3, decoded_tokens)
    once_gennerated_sequence = decoded_text
    print(f"once_gennerated_mask_sequence:{once_gennerated_sequence}")

    # variable_seq_prob=once_mask_token_probability[:,pre_link_len+1:-1-rever_link_len,:]
    topk_values, topk_indices = torch.topk(once_mask_token_probability, k=1)
    once_mask_token_probability[:, :pre_link_len + 2, :] = float("inf")
    once_mask_token_probability[:, -1 - rever_link_len:, :] = float("inf")
    """mask generate optimization"""

    mask_num = int(lowest_probs_mask_ratio * len(input_embedding[0]))
    mask_num = 1
    for i in range(iterative_optimize):
        for idx, m in enumerate(once_predict_seq_id[0, :].tolist()[1:-1]):
            if m in [0, 1, 2, 3]:
                once_predict_seq_id[0, idx + 1] = tokenizer.mask_token_id

        min_values, min_indices = torch.topk(topk_values.view(-1), k=mask_num, largest=False)
        min_indices_list = min_indices.tolist()

        once_predict_seq_id[0, min_indices_list] = tokenizer.mask_token_id

        # once_predict_seq_id=once_predict_seq_id.unsqueeze(dim=0)

        # mask_token=torch.tensor(tokenizer.mask_token_id , device="cuda")
        # once_predict_seq_id[min_indices]=mask_token
        mask_model_predict_logits = encode_model(once_predict_seq_id)[0]
        iterative_mask_token_probability = torch.softmax(mask_model_predict_logits, dim=2)
        once_mask_token_probability[:, min_indices, :] = iterative_mask_token_probability[:, min_indices, :]
        # variable_seq_prob = once_mask_token_probability[: , pre_link_len + 1:-1 - rever_link_len , :]
        topk_values, topk_indices = torch.topk(once_mask_token_probability, k=1)
        once_predict_seq_id[0, min_indices_list] = torch.argmax(mask_model_predict_logits, dim=2)[0, min_indices_list]

    decoded_tokens = tokenizer.convert_ids_to_tokens(once_predict_seq_id.view(-1))[1:-1]
    decoded_text = kmer2seq(3, decoded_tokens)
    once_gennerated_sequence = decoded_text

    def replace_start_end_with_strings(input_string, start_substring, end_substring):
        # 检查输入是否为空
        if not input_string or not start_substring or not end_substring:
            return input_string

        # 获取子串的长度
        start_len = len(start_substring)
        end_len = len(end_substring)

        # 替换开头
        input_string = start_substring + input_string[start_len:]

        # 替换结尾
        input_string = input_string[:-end_len] + end_substring

        return input_string

    once_gennerated_sequence = replace_start_end_with_strings(once_gennerated_sequence, pre_link, rever_link)
    print(f"once_gennerated_after_iterative_mask_sequence:{once_gennerated_sequence}")
    return once_gennerated_sequence

import copy
def mask_seq_genernate_nolinkers(input_embedding, decode_model, full_model, encode_model, tokenizer ,pre_link, rever_link, lowest_probs_mask_ratio=0.15,
                       iterative_optimize=3, top_k=3, top_p=0.8, temperature=1,):
    logits = decode_model(input_embedding)
    once_mask_token_probability = torch.softmax(logits, dim=2)
    # once_predict_seq_id = torch.argmax(once_mask_token_probability , dim=2)
    # logits[:, :, 0:4] = -100
    once_predict_seq_id = top_k_top_p_sampling(logits, top_k, top_p, temperature)

    def kmers_sliding_windows(seq, kmers=3):
        return " ".join([seq[i:i + kmers] for i in (range(len(seq) - kmers + 1))])

    once_predict_seq_id[0, 0] = tokenizer.cls_token_id
    once_predict_seq_id[0, -1] = tokenizer.sep_token_id



    ##print first generated sequence
    decoded_tokens = tokenizer.convert_ids_to_tokens(once_predict_seq_id.view(-1))[1:-1]
    decoded_text = kmer2seq(3, decoded_tokens)
    once_gennerated_sequence = decoded_text
    # print(f"once_gennerated_mask_sequence:{once_gennerated_sequence}")

    # variable_seq_prob=once_mask_token_probability[:,pre_link_len+1:-1-rever_link_len,:]
    topk_values, topk_indices = torch.topk(once_mask_token_probability, k=1)
    """mask generate optimization"""

    mask_num = int(lowest_probs_mask_ratio * len(input_embedding[0]))
    # mask_num = 1

    for idx, m in enumerate(once_predict_seq_id[0, :].tolist()[1:-1]):
        if m in [0, 1, 2, 3]:
            once_predict_seq_id[0, idx + 1] = tokenizer.mask_token_id
    mask_model_predict_probs=encode_model(once_predict_seq_id)[0]
    mask_model_predict_probs[:, :, 0:4] = -100
    mask_model_predict_indices= torch.argmax(mask_model_predict_probs,dim=2)
    mask_indices=torch.where(once_predict_seq_id == tokenizer.mask_token_id)[1]
    once_predict_seq_id[:,mask_indices]=mask_model_predict_indices[:,mask_indices]
    once_mask_token_probability[:, mask_indices, :] = torch.softmax(mask_model_predict_probs,dim=2)[:, mask_indices, :]
    decoded_tokens = tokenizer.convert_ids_to_tokens(once_predict_seq_id.view(-1))[1:-1]
    decoded_text = kmer2seq(3, decoded_tokens)
    once_gennerated_sequence = decoded_text
    # print(f"masked_gennerated_mask_sequence:{once_gennerated_sequence}")

    for i in range(iterative_optimize):


        min_values, min_indices = torch.topk(topk_values.view(-1), k=mask_num, largest=False)
        min_indices_list = min_indices.tolist()
        temp_list=copy.deepcopy(min_indices_list)
        for idx in temp_list:
            min_indices_list.extend([idx-1,idx+1])
        min_indices_list=list(set(min_indices_list))
        min_indices_list = [item for item in min_indices_list if item > -1 and item <once_predict_seq_id.size()[1]]

        once_predict_seq_id[0, min_indices_list] = tokenizer.mask_token_id

        # once_predict_seq_id=once_predict_seq_id.unsqueeze(dim=0)

        # mask_token=torch.tensor(tokenizer.mask_token_id , device="cuda")
        # once_predict_seq_id[min_indices]=mask_token
        mask_model_predict_logits = encode_model(once_predict_seq_id)[0]
        iterative_mask_token_probability = torch.softmax(mask_model_predict_logits, dim=2)
        once_mask_token_probability[:, min_indices, :] = iterative_mask_token_probability[:, min_indices, :]
        # variable_seq_prob = once_mask_token_probability[: , pre_link_len + 1:-1 - rever_link_len , :]
        topk_values, topk_indices = torch.topk(once_mask_token_probability, k=1)
        mask_model_predict_probs[:, :, 0:4] = -100
        once_predict_seq_id[0, min_indices_list] = torch.argmax(mask_model_predict_logits, dim=2)[0, min_indices_list]
        once_predict_seq_id[0, 0] = tokenizer.cls_token_id
        once_predict_seq_id[0, -1] = tokenizer.sep_token_id

    decoded_tokens = tokenizer.convert_ids_to_tokens(once_predict_seq_id.view(-1))[1:-1]
    decoded_text = kmer2seq(3, decoded_tokens)
    once_gennerated_sequence = decoded_text


    if pre_link and rever_link:
        final_gennerated_sequence = pre_link+once_gennerated_sequence+rever_link
    else:
        final_gennerated_sequence=once_gennerated_sequence

    return final_gennerated_sequence





if __name__ == "__main__":

    embedding_generation()
