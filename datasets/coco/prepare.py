import os
import json
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO


def save_as_json(basename, dataset):
    filename = os.path.join(os.path.dirname(__file__), basename)
    print("Saving %s ..." % filename)
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help="path to a directory containing COCO 2017 dataset."
    )
    args = parser.parse_args()

    for split in ['train', 'val']:
        coco = COCO(
            os.path.join(
                args.root,
                'annotations/instances_%s2017.json' % split
            )
        )

        count = 0
        ids = sorted(coco.imgs.keys())
        dataset = []
        for id in tqdm(ids):
            image_path = os.path.join(
                args.root,
                split + '2017',
                coco.loadImgs(id)[0]["file_name"]
            )
            anno = coco.loadAnns(coco.getAnnIds(id))
            boxes, classes = [], []
            for obj in anno:
                if obj['iscrowd'] == 0:
                    xmin, ymin, w, h = obj['bbox']
                    if w <=0 or h <= 0:
                        print("Skip an object with degenerate bbox (w=%.2f, h=%.2f)."
                              % (w, h))
                        continue
                    boxes.append([xmin, ymin, xmin + w, ymin + h])
                    classes.append(coco.getCatIds().index(obj['category_id']))
            dataset.append(
                {
                    'image': os.path.abspath(image_path),
                    'boxes': boxes,
                    'classes': classes,
                    'difficulties': [0 for _ in classes]
                }
            )
        save_as_json(split + '.json', dataset)


if __name__ == '__main__':
    main()
