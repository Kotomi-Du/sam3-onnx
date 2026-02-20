#!/usr/bin/env python3

import pathlib
import typing

import imgviz
import numpy as np
import onnxruntime
import PIL.Image
import torch
from loguru import logger
from numpy.typing import NDArray
from osam._models.yoloworld.clip import tokenize
from torchvision.transforms import v2
import os, sys
repo_root = os.path.abspath(r"C:\Users\yarudu\Documents\project\sam3-onnx\sam3")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from infer_torch import get_replace_freqs_cis
from sam3.model.sam3_image import Sam3Image  # type: ignore[unresolved-import]
from sam3.model.sam3_image_processor import (  # type: ignore[unresolved-import]
    Sam3Processor,
)
from sam3.model_builder import build_sam3_image_model  # type: ignore[unresolved-import]

from export_onnx import _ImageEncoder, _LanguageEncoder, _Decoder
from pathlib import Path
import openvino as ov


def convert_onnx_to_ov(onnx_path, ov_model_path):
    onnx_path = "models/sam3_decoder.onnx"
    core = ov.Core()
    ov_model = core.read_model(str(onnx_path))

    # Save OpenVINO IR
    ov_model_path = Path(onnx_path.replace(".onnx", ".xml"))
    ov.save_model(ov_model, str(ov_model_path))
    print(f"✓ OpenVINO model saved to {ov_model_path}")



def _export_image_encoder(processor: Sam3Processor, image: PIL.Image.Image
) -> tuple[list[NDArray], list[NDArray]]:
    image = image.resize((1008, 1008), resample=PIL.Image.BILINEAR)

    ov_model_path: pathlib.Path = pathlib.Path("models/sam3_image_encoder.xml")
    if ov_model_path.exists():
        logger.debug("onnx model already exists, skip export: {!r}", str(ov_model_path))
    else:
        processor.model = processor.model.to("cpu")
        encoder: _ImageEncoder = _ImageEncoder(processor=processor)
        encoder.eval()  # Set encoder to eval mode
        input_image: torch.Tensor = v2.functional.to_image(image).to("cpu")

        ov_model = ov.convert_model(encoder, example_input=input_image)
        ov.save_model(ov_model, ov_model_path)
        print(f"✓ OpenVINO model saved to {ov_model_path}")

    core = ov.Core()
    model_image = core.read_model(ov_model_path)
    compiled_image = core.compile_model(model_image, "CPU")
    image_input = np.asarray(image.resize((1008, 1008))).transpose(2, 0, 1)
    output = compiled_image({"image": image_input}) 
    return output


def _export_language_encoder(processor: Sam3Processor) -> list[NDArray]:
    tokens = tokenize(texts=["person"], context_length=32)

    ov_model_path: pathlib.Path = pathlib.Path("models/sam3_language_encoder.xml")
    if ov_model_path.exists():
        logger.debug("onnx model already exists, skip export: {!r}", str(ov_model_path))
    else:
        processor.model = processor.model.to("cpu")
        encoder: _LanguageEncoder = _LanguageEncoder(processor=processor)
        tokens_input: torch.Tensor = torch.from_numpy(tokens).to("cpu")

        ov_model = ov.convert_model(encoder, example_input=tokens_input)
        ov.save_model(ov_model, ov_model_path)
        print(f"✓ OpenVINO model saved to {ov_model_path}")

    core = ov.Core()
    model_language = core.read_model(ov_model_path)
    compiled_language = core.compile_model(model_language, "CPU")
    output = compiled_language({"tokens": tokens}) 
    return output


def _export_decoder(
    original_height: int,
    original_width: int,
    vision_pos_enc_0: NDArray,
    vision_pos_enc_1: NDArray,
    vision_pos_enc_2: NDArray,
    backbone_fpn_0: NDArray,
    backbone_fpn_1: NDArray,
    backbone_fpn_2: NDArray,
    language_mask: NDArray,
    language_features: NDArray,
    language_embeds: NDArray,
    box_coords: NDArray,
    box_labels: NDArray,
    box_masks: NDArray,
) -> list[NDArray]:
    ov_model_path: pathlib.Path = pathlib.Path("models/sam3_decoder.xml")
    if ov_model_path.exists():
        logger.debug("onnx model already exists, skip export: {!r}", str(ov_model_path))
    else:
        decoder: _Decoder = _Decoder()

        ov_model = ov.convert_model(decoder, example_input={
        "original_height": np.array(original_height, dtype=np.int64),
        "original_width": np.array(original_width, dtype=np.int64),
        "backbone_fpn_0": backbone_fpn_0,
        "backbone_fpn_1": backbone_fpn_1,
        "backbone_fpn_2": backbone_fpn_2,
        "vision_pos_enc_0": vision_pos_enc_0,
        "vision_pos_enc_1": vision_pos_enc_1,
        "vision_pos_enc_2": vision_pos_enc_2,
        "language_mask": language_mask,
        "language_features": language_features,
        "language_embeds": language_embeds,
        "box_coords": box_coords,
        "box_labels": box_labels,
        "box_masks": box_masks,
    })
        ov.save_model(ov_model, ov_model_path)
        print(f"✓ OpenVINO model saved to {ov_model_path}")


    core = ov.Core()
    model_decoder = core.read_model(ov_model_path)
    compiled_decoder = core.compile_model(model_decoder, "CPU")
    output = compiled_decoder( {"original_height": np.array(original_height, dtype=np.int64),
        "original_width": np.array(original_width, dtype=np.int64),
         "backbone_fpn_0": backbone_fpn_0,
        "backbone_fpn_1": backbone_fpn_1,
        "backbone_fpn_2": backbone_fpn_2,
        #"vision_pos_enc_0": vision_pos_enc_0,
        #"vision_pos_enc_1": vision_pos_enc_1,
        "vision_pos_enc_2": vision_pos_enc_2,
        "language_mask": language_mask,
        "language_features": language_features,
        # "language_embeds": language_embeds,
        "box_coords": box_coords,
        "box_labels": box_labels,
        "box_masks": box_masks,
    }) 
    return output

def main():
    model: Sam3Image = build_sam3_image_model()
    # Set model to eval mode for inference
    model.eval()
    # Disable gradients for all parameters
    for param in model.parameters():
        param.requires_grad_(False)
    get_replace_freqs_cis(model)
    processor: Sam3Processor = Sam3Processor(model, device="cpu")

    image: PIL.Image.Image = PIL.Image.open("images/bus.jpg")

    # image_encoder {{
    # state = processor.set_image(image)

    image_output = _export_image_encoder(processor, image)

    vision_pos_enc: list[NDArray] = [
        image_output[0], #["vision_pos_enc_0"],
        image_output[1], #["vision_pos_enc_1"],
        image_output[2], #["vision_pos_enc_2"],
    ]
    backbone_fpn: list[NDArray] = [
        image_output[3], #["backbone_fpn_0"],
        image_output[4], #["backbone_fpn_1"],
        image_output[5], #["backbone_fpn_2"],
    ]

    language_output = _export_language_encoder(processor)
    language_mask: NDArray = language_output[0] #["text_attention_mask"]
    language_features: NDArray = language_output[1] #["text_memory"]
    language_embeds: NDArray = language_output[2] #["text_embeds"]

    box_coords = np.array([[[0.1620, 0.4010, 0.0640, 0.0180]]], dtype=np.float32)
    box_labels = np.array([[1]], dtype=np.int64)
    box_masks = np.array([[True]], dtype=np.bool_)

    final_output = _export_decoder(
        original_height=image.height,
        original_width=image.width,
        vision_pos_enc_0=vision_pos_enc[0],
        vision_pos_enc_1=vision_pos_enc[1],
        vision_pos_enc_2=vision_pos_enc[2],
        backbone_fpn_0=backbone_fpn[0],
        backbone_fpn_1=backbone_fpn[1],
        backbone_fpn_2=backbone_fpn[2],
        language_mask=language_mask,
        language_features=language_features,
        language_embeds=language_embeds,
        box_coords=box_coords,
        box_labels=box_labels,
        box_masks=box_masks,
    )

    boxes: NDArray = final_output[0] #["boxes"]
    scores: NDArray = final_output[1] #["scores"]
    masks: NDArray = final_output[2] #["masks"]


    
    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks[:, 0, :, :],
        bboxes=boxes[:, [1, 0, 3, 2]],
        labels=np.arange(len(boxes)) + 1,
        captions=[f"{s:.2f}" for s in scores],
    )
    
    # Save result to images folder
    output_dir = pathlib.Path("images")
    output_dir.mkdir(exist_ok=True)
    output_filename = f"export_result_openvino.jpg"
    output_path = output_dir / output_filename
    PIL.Image.fromarray(viz).save(output_path)
    logger.info("saved result to: {}", output_path)
    



if __name__ == "__main__":
    main()
