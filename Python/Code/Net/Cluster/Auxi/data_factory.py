from Code.Net.Cluster.Auxi.data_loader import UEAloader
from Code.Net.Cluster.Auxi.uea import collate_fn
from torch.utils.data import DataLoader

# 定义的数据预加载程序，如果需要加载自己的数据，需要去data_loader.py以重写dataloader以适合您的数据

def data_provider(args, flag):
    Data = UEAloader

    # 用于判断是否是训练/测试
    if flag == 'TEST':  # 加载测试数据    #
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True # 用于训练和验证
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid

    # 用于判断加载类型
    data_set = Data(
        root_path=args.root_path+args.model_id+'/',
        lab_rep=args.lab_rep,
        flag=flag,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)     # #define some limits to collate pieces of data into batches
    )                               # collate_fn 处理动态变化的批处理长度，把batch长度调整一致
                                    # 同时padding_mask矩阵也是这个函数生成的
    return data_set, data_loader
