"""
run e.g. with: 
- python run_ocr.py -i "SampleData" -e "jpg"

or using the layout model:
- python run_ocr.py -i "SampleData" -e "jpg" -m "Layout"
"""

import os
import cv2
import sys
import pyewts
import argparse
from glob import glob
from tqdm import tqdm
from Lib.Data import OCRStatus
from natsort import natsorted
from Lib.Utils import create_dir, get_file_name, read_line_model_config, read_layout_model_config
from Lib.Config import init_line_model, init_layout_model, init_ocr_model
from Lib.Inference import OCRPipeline
from Lib.Exporter import PageXMLExporter, JsonExporter, TextExporter


line_model_config_file = init_line_model()
layout_model_config_file = init_layout_model()

line_model_config = read_line_model_config(line_model_config_file)
layout_model_config = read_layout_model_config(layout_model_config_file)

# use a pre-registered model or point to a config file (*.config) of a compatible downloaded model
ocr_model_config = init_ocr_model("Namgyal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=False, default="Output")
    parser.add_argument("-e", "--file_extension", type=str, required=False, default="jpg")
    parser.add_argument("-m", "--mode", choices=["Line", "Layout"], default="Layout")
    parser.add_argument("-k", "--k_factor", type=float, required=False, default=1.7)
    parser.add_argument("-b", "--bbox", type=float, required=False, default=2.5)
    parser.add_argument("-s", "--save_format", choices=["xml", "json", "text"], default="text")
    parser.add_argument("-f", "--output_format", choices=["wylie, unciode"], required=False, default="unicode")

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    file_ext = args.file_extension
    mode = args.mode
    k_factor = float(args.k_factor)
    bbox_tolerance = float(args.bbox)
    save_format = args.save_format
    output_format = args.output_format

    if not os.path.isdir(input_dir):
        print("ERROR: Input dir is not a valid directory")
        sys.exit(1)
    
    images = natsorted(glob(f"{input_dir}/*.{file_ext}"))
    print(f"Images: {len(images)}")

    dir_name = os.path.basename(input_dir)
    output_dir = os.path.join(output_dir, dir_name)
    create_dir(output_dir)

    converter = pyewts.pyewts()

    if save_format == "xml":
        exporter = PageXMLExporter(output_dir)
    elif save_format == "json":
        exporter = JsonExporter(output_dir)
    else:
        exporter = TextExporter(output_dir)

    if mode == "Line":
        ocr_pipeline = OCRPipeline(ocr_model_config, line_model_config, output_dir)

    else:
        ocr_pipeline = OCRPipeline(ocr_model_config, layout_model_config, output_dir)

    for idx, image_path in tqdm(enumerate(images), total=len(images)):
        image_name = get_file_name(image_path)
        img = cv2.imread(image_path)
        status, ocr_result = ocr_pipeline.run_ocr(img, image_name, k_factor=k_factor, bbox_tolerance=bbox_tolerance)

        if status == OCRStatus.SUCCESS:
            if len(ocr_result.lines) > 0:
                if output_format == "unicode":
                    text = [converter.toUnicode(x) for x in ocr_result.text]
                    if isinstance(exporter, TextExporter):
                        exporter.export_lines(img, image_name, ocr_result.lines, ocr_result.text)
                    else:
                        exporter.export_lines(img, image_name, ocr_result.lines, text, angle=ocr_result.angle)
                else:
                    if isinstance(exporter, TextExporter):
                        exporter.export_lines(img, image_name, ocr_result.lines, ocr_result.text)
                    else:
                        exporter.export_lines(img, image_name, ocr_result.lines, ocr_result.text, angle=ocr_result.angle)
