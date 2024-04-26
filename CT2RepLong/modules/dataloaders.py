import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader


class R2DataLoader(DataLoader):
    def __init__(self, args, dataset, tokenizer, split, shuffle):
        self.args = args
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        self.dataset = dataset
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        #print("im")
        images_id, images, reports_ids, reports_masks, seq_lengths,reports,mask,context_len,image_2 = zip(*data)
        image_2=torch.stack(image_2, 0)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)
        max_len=max(context_len)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks
        targets_contexts = np.zeros((len(reports), max_len), dtype=int)
        contexts_masks = np.zeros((len(reports), max_len), dtype=int)

        for i, context in enumerate(reports):
            targets_contexts[i, :len(context)] = context

        for i, masks in enumerate(mask):
            contexts_masks[i, :len(masks)] = masks
        return images_id, images,image_2, torch.LongTensor(targets), torch.FloatTensor(targets_masks),torch.LongTensor(targets_contexts), torch.FloatTensor(contexts_masks)


