import sys
import os
import json
import argparse
import datetime
import numpy as np
import skimage.draw

import pickle
import tensorflow as tf
from icecream import ic
from logging import DEBUG, ERROR, INFO

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
FLWR_DIR = os.path.join(ROOT_DIR, 'src')
MASKRCNN_DIR = os.path.join(ROOT_DIR, 'maskrcnn')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')
# LOCAL_MODEL_DIR = os.path.join(ROOT_DIR, 'model', '00000.weights')

# Import Flower
sys.path.append(FLWR_DIR)
import flwr as fl
from flwr.common.logger import log

# Import Mask RCNN
sys.path.append(MASKRCNN_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
WEIGHTS_SAVE_PATH= os.path.join(WEIGHTS_DIR, '00000.weights')

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class BalloonConfig(Config):
    NAME = "balloon"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + balloon
    # STEPS_PER_EPOCH = 100
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("balloon", 1, "balloon")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def get_length(self):
        return len(self.image_info)


def get_datasets():
    # Training dataset
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    return dataset_train, dataset_val




def train(model, dataset_train, dataset_val):
    """Train the model."""

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.

    print("Training network heads")
    history = model.train(dataset_train,
                dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads'),
                # layers='rpn_model')
                # layers='all')


    return history


client_dict = {}
client_dict['id'] = 'client0'


if __name__ == "__main__":

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    ### Add Arguments
    # Flower
    parser.add_argument('--ip', type=str, required=True, help='Enter Server IP')
    parser.add_argument('--port', type=str, default='8080', help='Enter Server Port, default: 8080')

    # MaskRCNN
    parser.add_argument('--command', type=str, default='train', help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False, default=os.path.join(ROOT_DIR, 'datasets', 'balloon'), help='Directory of the Balloon dataset')
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--init', default=False, action='store_true', help='Create 00000.weight file using coco pretrained weights')

    # Parse
    args = parser.parse_args()

    # Configuration for training 
    config = BalloonConfig()

    # Prepare and get datasets
    dataset_train, dataset_val = get_datasets()
    
    client_dict['num_test'] = dataset_val.get_length()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    if args.init:
        # Load pretrained coco model
        # Download weights file
        if not os.path.exists(COCO_WEIGHTS_PATH):
            utils.download_trained_weights(COCO_WEIGHTS_PATH)

        # Load weights
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
            "mrcnn_class_logits",
            "mrcnn_bbox_fc",
            "mrcnn_bbox",
            "mrcnn_mask",
            "rpn_model"  # because anchor's ratio has been changed
        ])

        # Get weights and save them
        weights = model.m_get_weights()

        # Save weights
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)

        try:
            log(INFO, f'save weights to {WEIGHTS_SAVE_PATH}')
            fs = open(WEIGHTS_SAVE_PATH, '+wb')
            pickle.dump(weights, fs )
            fs.close()
        except:
            log(ERROR, f'Error when trying to saving weights to {WEIGHTS_SAVE_PATH}')

        # Leave
        sys.exit()


    # Define Flower client
    class MaskRCNNClient(fl.client.NumPyClient):

        def get_parameters(self):  # type: ignore
            return model.m_get_weights()            

        def fit(self, parameters, config):  # type: ignore
            model.m_set_weights(parameters)
            # model.fit(x_train, y_train, epochs=3, batch_size=32, steps_per_epoch=3)
            history_train = train(
                model,
                dataset_train,
                dataset_val
            )

            client_dict['num_train'] = dataset_train.get_length()
            client_dict['train_history'] = str(history_train[0].history)

            # # Do a evaluation on the test set using the local model
            # Values inside the list correspond to: loss, rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss
            client_dict['test_local'] = str(model.eval(dataset_val))


            return model.m_get_weights(), client_dict['num_train']

        def evaluate(self, parameters, config):  # type: ignore
            model.m_set_weights(parameters)

            # get hash of model
            client_dict['hashcode'] = model.get_hash(encoding='md5')
   
            # # Do a evaluation on the test set using the global model
            # Values inside the list correspond to: loss, rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss
            history_test = model.eval(dataset_val)
            client_dict['test_global'] = str(history_test)

            # In Flower, release 0.14 it will be possible to return dicts
            # https://github.com/adap/flower/issues/632
            # https://flower.dev/docs/changelog.html
            serialized_dict = json.dumps(client_dict, skipkeys = True)

            # Save weights
            try:
                log(DEBUG, f'save weights to {WEIGHTS_SAVE_PATH}')
                fs = open(WEIGHTS_SAVE_PATH, '+wb')
                pickle.dump(parameters, fs)
                fs.close()
            except:
                log(ERROR, f'Error when trying to saving weights to {WEIGHTS_SAVE_PATH}')

            return client_dict['num_test'], None, None, {"custom_metric": serialized_dict}
            # return len(x_test), loss, accuracy, {"custom_metric": serialized_dict}

    
    # Start Flower client
    fl.client.start_numpy_client(f"{args.ip}:{args.port}", client=MaskRCNNClient())