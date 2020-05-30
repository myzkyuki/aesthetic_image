import os
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

from absl import logging

TRAIN_RATE = 0.97

logging.set_verbosity(logging.INFO)


def convert(annotation_path: str, image_dir: str,
            train_rate: float = TRAIN_RATE):
    annot_df = pd.read_table(annotation_path, sep=' ', header=None,
                             names=['Index', 'Image ID',
                                    '1', '2', '3', '4', '5',
                                    '6', '7', '8', '9', '10',
                                    'Semantic tag ID 1', 'Semantic tag ID 2',
                                    'Challenge ID'],
                             index_col=0)
    logging.info(f'Load {len(annot_df)} records from {annotation_path}.')

    # Check is file readable
    annot_df['file_readable'] = False
    for i, (index, row) in enumerate(annot_df.iterrows()):
        image_id = row['Image ID']
        image_path = os.path.join(image_dir, f'{image_id}.jpg')
        try:
            img = Image.open(image_path)
            img.getdata()
            annot_df.at[index, 'file_readable'] = True
        except Exception:
            logging.warning(f'{image_path} is not readable')

        if (i + 1) % (len(annot_df) // 50) == 0:
            logging.info(f'{i+1} / {len(annot_df)} images are checked.')

    annot_df = annot_df[annot_df['file_readable']]
    logging.info(f'Validate {len(annot_df)} readable images.')

    # Normalize scores
    annot_df['num_votes'] = annot_df[map(str, range(1, 11))].sum(axis=1)
    for i in range(1, 11):
        annot_df[str(i)] /= annot_df['num_votes']

    logging.info(f'Normalize scores')

    # Split to train and validation dataset
    annot_df = annot_df.sample(frac=1, random_state=1)
    num_train = int(len(annot_df) * train_rate)
    train_df = annot_df[:num_train]
    validation_df = annot_df[num_train:]

    logging.info(f'Split to {len(train_df)} train images and '
                 f'{len(validation_df)} validation images.')

    # Save csv
    dir_path = os.path.dirname(annotation_path)
    train_path = os.path.join(dir_path, 'AVA_train.csv')
    validation_path = os.path.join(dir_path, 'AVA_validation.csv')
    train_df.to_csv(train_path)
    validation_df.to_csv(validation_path)
    logging.info(f'Save train data to {train_path} and '
                 f'validation data to {validation_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, required=True,
                        help='/path/to/AVA_dataset/AVA.txt')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='/path/to/images')
    parser.add_argument('--train_rate', type=float, default=TRAIN_RATE,
                        help='Train images rate')
    args = parser.parse_args()

    convert(args.annotation_path, args.image_dir, args.train_rate)

