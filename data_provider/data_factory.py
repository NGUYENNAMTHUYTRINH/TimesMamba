from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_Solar,
    Dataset_PEMS,
    Dataset_Pred,
)
from torch.utils.data import DataLoader

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "Solar": Dataset_Solar,
    "PEMS": Dataset_PEMS,
    "custom": Dataset_Custom,
    "weather": Dataset_Custom,
}


def data_provider(args, flag):
    # If requesting predictions, use Dataset_Pred.
    # If the provided data_path appears to be a split file (contains '_train', '_val', or '_test'),
    # use Dataset_Custom which expects pre-split CSVs with date+features+target columns.
    if flag == "pred":
        Data = Dataset_Pred
    else:
        dp = args.data_path.lower() if hasattr(args, 'data_path') and args.data_path else ''
        if any(x in dp for x in ['_train', '_val', '_test']):
            Data = Dataset_Custom
        else:
            Data = data_dict[args.data]

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=1 if args.embed == "timeF" else 0,
        freq=args.freq,
    )
    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True if flag == "train" else False,
        num_workers=args.num_workers,
    )

    return data_set, data_loader
