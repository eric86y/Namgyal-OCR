"""
A simple interface to generate a OCR training dataset from PageXML data:

example usage: 

> python run_data_generation.py -d Eric-23xd/Namgyal-OCR-Annotations

"""

import os
import argparse
from glob import glob
from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile
from natsort import natsorted
from huggingface_hub import snapshot_download
from Lib.Utils import create_dir, create_dataset


def download_hf_dataset(
    xml_datasat_id: str = "Eric-23xd/Namgyal-OCR-Annotations",
    annotations_output_path: str = "Data/Namgyal/Annotations",
):
    create_dir(annotations_output_path)

    try:
        data_dir = snapshot_download(
            repo_id=xml_datasat_id, repo_type="dataset", cache_dir="Data"
        )
    except BaseException as e:
        print(f"Failed to download dataset {xml_datasat_id} from HuggingFace: {e}")

    zip_file = f"{data_dir}/data.zip"

    if os.path.isfile(zip_file):
        try:
            with ZipFile(zip_file, "r") as zipFile:
                zipFile.extractall(path=annotations_output_path)

        except FileExistsError as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=False, default="Output")
    parser.add_argument("-i", "--iterations", type=int, required=False, default=6)

    args = parser.parse_args()
    dataset_id = args.dataset
    output_dir = args.output
    kernel_iterations = args.iterations

    download_dir = os.path.join(output_dir, "Annotations")
    create_dir(download_dir)
    download_hf_dataset(xml_datasat_id=dataset_id, annotations_output_path=download_dir)

    dataset_img_out = os.path.join(output_dir, "Dataset", "lines")
    dataset_transcriptions_out = os.path.join(output_dir, "Dataset", "transcriptions")
    create_dir(dataset_img_out)
    create_dir(dataset_transcriptions_out)

    for volume_dir in Path(download_dir).iterdir():
        images = natsorted(glob(f"{volume_dir}/*.jpg"))
        xml_files = natsorted(glob(f"{volume_dir}/page/*.xml"))

        print(
            f"{volume_dir} => Images: {len(images)}, XML-Annotations: {len(xml_files)}"
        )

        for image, annotation in tqdm(zip(images, xml_files), total=len(xml_files)):
            create_dataset(
                image, annotation, dataset_img_out, dataset_transcriptions_out
            )
