import imghdr
from datetime import datetime
import os
from shutil import copyfile
#from .definitions import camera_id_with_train # PATH_CONTAINER_IMAGES, PATH_IMAGES,
from pydantic.main import BaseModel

from .definitions import PATH_IMAGES, PATH_CONTAINER_IMAGES, camera_id_with_train


class FileNameMetadata(BaseModel):
    """ Holds information about image metadata from filename"""
    filename: str
    production_time: datetime
    pick_id: int
    cam_id: str
    crane_id: str


def extract_filename_metadata(filename: str) -> FileNameMetadata:
    """ Extracts metadata from filename string and returns it as model

    Example filename: '2021-03-08_00-00-40_pick-017482_cam-Par1CarB_crane-PR13.jpg'
    can be split by file ending ('.jpg'), '_' and keywords: 'pick', 'cam', 'crane'
    to: '{production_time}_pick-{pick_id}_cam-{cam_id}_crane-{crane_id}{file ending}'

    Args:
        filename: name of image file
    """

    filename = filename.split(".jpg")[0]

    split_name = filename.split("_")

    time_string = "_".join(split_name[:2])
    production_time = datetime.strptime(time_string, "%Y-%m-%d_%H-%M-%S")

    pick_id = int(split_name[2].split("pick-", 1)[1])

    cam_id = split_name[3].split("cam-", 1)[1]

    crane_id = "_".join(split_name[4:]).split("crane-", 1)[1]

    return FileNameMetadata(
        cam_id=cam_id,
        crane_id=crane_id,
        filename=filename,
        pick_id=pick_id,
        production_time=production_time
    )


def extract_container_images():
    """ Extracts images with container.
    The filenames of images showing the buckets all include the term *car*
    the filenames images of arriving trains include
    *Pr{number}Pos* and *Pr{number}Ant*
    """
    # if not os.path.exists(PATH_CONTAINER_IMAGES):
    #     os.mkdir(PATH_CONTAINER_IMAGES)

    if not os.path.exists(PATH_IMAGES):
        os.mkdir(PATH_IMAGES)

    # todo: implement DVC
    images = [f for f in os.listdir(PATH_CONTAINER_IMAGES) if imghdr.what(os.path.join(PATH_CONTAINER_IMAGES, f)) is not None]

    for image in images:
        metadata = extract_filename_metadata(image)
        if metadata.cam_id in camera_id_with_train:
            new_image_path = os.path.join(PATH_IMAGES, image)
            old_image_path = os.path.join(PATH_CONTAINER_IMAGES, image)
            copyfile(old_image_path, new_image_path)


if __name__ == "__main__":
    extract_container_images()
