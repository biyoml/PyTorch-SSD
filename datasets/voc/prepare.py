import os
import json
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    image_width = int(root.find('size').find('width').text)
    image_height = int(root.find('size').find('height').text)
    boxes, classes, difficulties = [], [], []
    for object in root.iter('object'):
        bndbox = object.find('bndbox')
        xmin = int(bndbox.find('xmin').text) - 1
        ymin = int(bndbox.find('ymin').text) - 1
        xmax = int(bndbox.find('xmax').text) - 1
        ymax = int(bndbox.find('ymax').text) - 1
        boxes.append([xmin, ymin, xmax, ymax])

        label = object.find('name').text.lower().strip()
        classes.append(label)

        difficulty = int(object.find('difficult').text == '1')
        difficulties.append(difficulty)
    return boxes, classes, difficulties


def save_as_json(basename, dataset):
    filename = os.path.join(os.path.dirname(__file__), basename)
    print("Saving %s ..." % filename)
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, required=True, help="path to VOCdevkit/")
    args = parser.parse_args()

    class_names = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    ]

    paths = {
        2007: os.path.join(args.root, 'VOC2007/'),
        2012: os.path.join(args.root, 'VOC2012/')
    }

    # Training data
    train_dataset = []
    for year, path in paths.items():
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = [line.strip() for line in f.readlines()]
        for id in tqdm(ids, desc="train %d" % year):
            image_path = os.path.join(path, 'JPEGImages', id + '.jpg')
            annotation_path = os.path.join(path, 'Annotations', id + '.xml')
            boxes, classes, difficulties = parse_annotation(annotation_path)
            classes = [class_names.index(c) for c in classes]
            train_dataset.append(
                {
                    'image': os.path.abspath(image_path),
                    'boxes': boxes,
                    'classes': classes,
                    'difficulties': difficulties
                }
            )
    save_as_json('train.json', train_dataset)

    # Validation data
    val_dataset = []
    voc07_path = paths[2007]
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = [line.strip() for line in f.readlines()]
    for id in tqdm(ids, desc="val 2007"):
        image_path = os.path.join(voc07_path, 'JPEGImages', id + '.jpg')
        annotation_path = os.path.join(voc07_path, 'Annotations', id + '.xml')
        boxes, classes, difficulties = parse_annotation(annotation_path)
        classes = [class_names.index(c) for c in classes]
        val_dataset.append(
            {
                'image': os.path.abspath(image_path),
                'boxes': boxes,
                'classes': classes,
                'difficulties': difficulties
            }
        )
    save_as_json('val.json', val_dataset)

    print("Complete.")


if __name__ == '__main__':
    main()
