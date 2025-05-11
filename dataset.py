import os
import random                            # <— make sure to import this
from PIL import Image
from torch.utils.data import Dataset

class SketchPhotoDataset(Dataset):
    def __init__(self, root_dir, transform):
        img_root = os.path.join(root_dir, "photo")
        sk_root  = os.path.join(root_dir, "sketch")
        self.transform = transform

        # build per-class image & sketch lists
        self.images  = {
            c: sorted(os.path.join(img_root, c, f)
                      for f in os.listdir(os.path.join(img_root, c)))
            for c in os.listdir(img_root)
            if os.path.isdir(os.path.join(img_root, c))
        }
        self.sketches = {
            c: sorted(os.path.join(sk_root, c, f)
                      for f in os.listdir(os.path.join(sk_root, c)))
            for c in os.listdir(sk_root)
            if os.path.isdir(os.path.join(sk_root, c))
        }

        # keep only classes present in both
        self.classes = sorted(set(self.images) & set(self.sketches))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # NEW: for each sketch, store (sketch_path, label)
        self.sketch_list = []
        for c in self.classes:
            idx = self.class_to_idx[c]
            for sp in self.sketches[c]:
                self.sketch_list.append((sp, idx))

    def __len__(self):
        # return number of sketches (one training sample per sketch)
        return len(self.sketch_list)

    def __getitem__(self, idx):
        sk_path, label = self.sketch_list[idx]

        # sample a positive photo from the same class
        pos_candidates = self.images[self.classes[label]]
        pos_path = random.choice(pos_candidates)

        # sample a negative photo from a different class
        neg_classes = [c for c in self.classes if c != self.classes[label]]
        neg_cls = random.choice(neg_classes)
        neg_path = random.choice(self.images[neg_cls])

        # load & transform
        sk = Image.open(sk_path).convert("RGB")
        im = Image.open(pos_path).convert("RGB")
        neg = Image.open(neg_path).convert("RGB")

        return (
            self.transform(sk),
            self.transform(im),
            # self.transform(neg),
            label
        )
