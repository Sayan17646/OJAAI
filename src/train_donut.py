"""
train_donut.py — PyTorch & Hugging Face Seq2Seq fine-tuning script for Donut.

Used to fine-tune naver-clova-ix/donut-base on handwritten prescriptions to extract
structured JSON fields directly from prescription images.

Instructions for running:
  1. Prepare a metadata.jsonl file alongside images:
     {"file_name": "image_1.png", "ground_truth": "{\"doctor_reg\": \"...\", \"medications\": [...]}"}
  2. Run: python -m src.train_donut --dataset_path ./data/training --output_dir ./models/donut-rx
"""

from __future__ import annotations

import os
import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Allows metadata/label validation on machines without Torch.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

if TYPE_CHECKING:
    from transformers import DonutProcessor, VisionEncoderDecoderModel

from src.donut_dataset import (
    compact_ground_truth,
    ground_truth_from_entry,
    iter_sample_summaries,
    load_metadata,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_START_TOKEN = "<s_rx>"
EOS_TOKEN = "</s>"

# --- XML/Sequence Conversion Helper ---

def json_to_donut_sequence(obj: Any) -> str:
    """
    Recursively convert a JSON dict/list/value to Donut's XML-style sequence format.
    Example:
      {"doctor_reg": "MCI-4819", "medications": [{"drug_name": "Amlodipine"}]} ->
      "<s_doctor_reg>MCI-4819</s_doctor_reg><s_medications><s_el><s_drug_name>Amlodipine</s_drug_name></s_el></s_medications>"
    """
    if isinstance(obj, dict):
        result = ""
        for k, v in obj.items():
            v = compact_ground_truth(v)
            if v in (None, "", [], {}):
                continue
            result += f"<s_{k}>"
            result += json_to_donut_sequence(v)
            result += f"</s_{k}>"
        return result
    elif isinstance(obj, list):
        result = ""
        for item in obj:
            item = compact_ground_truth(item)
            if item in (None, "", [], {}):
                continue
            # Wrap list elements in an element tag <s_el> ... </s_el>
            result += "<s_el>"
            result += json_to_donut_sequence(item)
            result += "</s_el>"
        return result
    else:
        # Primitive value
        if obj is None:
            return ""
        return str(obj)


def build_target_sequence(ground_truth: dict[str, Any]) -> str:
    """Build the exact Donut decoder target for one structured annotation."""
    compacted = compact_ground_truth(ground_truth)
    body = json_to_donut_sequence(compacted)
    if not body.strip():
        raise ValueError("Cannot build Donut target from empty ground_truth")
    return f"{TASK_START_TOKEN}{body}{EOS_TOKEN}"


def decoded_labels_to_text(labels: torch.Tensor, tokenizer, ignore_id: int = -100) -> str:
    """Decode a label tensor for human inspection before training."""
    labels = labels.detach().clone()
    labels[labels == ignore_id] = tokenizer.pad_token_id
    return tokenizer.decode(labels, skip_special_tokens=False).replace(tokenizer.pad_token or "", "").strip()


# --- Custom Dataset Class ---

class PrescriptionDataset(Dataset):
    """
    Custom PyTorch Dataset for loading prescription images and converting
    ground truth JSON targets into tokenized labels for Donut.
    """
    def __init__(
        self,
        dataset_path: str,
        processor: DonutProcessor,
        max_length: int = 512,
        split: str = "train",
        ignore_id: int = -100,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id

        metadata_file = self.dataset_path / "metadata.jsonl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

        # Load and validate every metadata record before training can start.
        self.samples = load_metadata(metadata_file)

        logger.info(f"Loaded {len(self.samples)} samples for split={split} from {dataset_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if torch is None or Image is None:
            raise ImportError("Dataset item loading requires torch and Pillow.")
        sample = self.samples[idx]

        # Load image
        img_path = self.dataset_path / sample["file_name"]
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # Process image pixel values
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze(0)  # [channels, height, width]

        # Format ground truth target text sequence
        gt = ground_truth_from_entry(sample, f"sample[{idx}]")
        target_seq = build_target_sequence(gt)

        # Tokenize labels
        labels = self.processor.tokenizer(
            target_seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Clone labels for loss calculation: replace padding token IDs with ignore_id
        # so Trainer ignores them in loss computation
        labels_for_loss = labels.clone()
        labels_for_loss[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id

        return {
            "pixel_values": pixel_values,
            "labels": labels_for_loss,
        }

    def target_sequence(self, idx: int) -> str:
        """Return the raw target sequence for debugging and validation."""
        gt = ground_truth_from_entry(self.samples[idx], f"sample[{idx}]")
        return build_target_sequence(gt)


# --- Training Script Entrypoint ---

def train(args):
    if torch is None:
        raise ImportError("Training requires torch. Install project dependencies before running Donut fine-tuning.")
    if Image is None:
        raise ImportError("Training requires Pillow. Install project dependencies before running Donut fine-tuning.")
    from transformers import (
        DonutProcessor,
        VisionEncoderDecoderModel,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    # Load processor and model
    logger.info(f"Loading pre-trained processor/model: {args.base_model}")
    processor = DonutProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    # Configure special tokens for prescription keys
    # These tokens are needed in vocabulary so they don't get split into subwords.
    special_tokens = [
        TASK_START_TOKEN, EOS_TOKEN,
        "<s_doctor_name>", "</s_doctor_name>",
        "<s_doctor_reg>", "</s_doctor_reg>",
        "<s_clinic_name>", "</s_clinic_name>",
        "<s_patient_name>", "</s_patient_name>",
        "<s_patient_age>", "</s_patient_age>",
        "<s_patient_gender>", "</s_patient_gender>",
        "<s_prescription_date>", "</s_prescription_date>",
        "<s_diagnosis>", "</s_diagnosis>",
        "<s_medications>", "</s_medications>",
        "<s_el>", "</s_el>",
        "<s_drug_name>", "</s_drug_name>",
        "<s_dosage_value>", "</s_dosage_value>",
        "<s_dosage_unit>", "</s_dosage_unit>",
        "<s_frequency>", "</s_frequency>",
        "<s_freq_per_day>", "</s_freq_per_day>",
        "<s_duration_days>", "</s_duration_days>",
        "<s_route>", "</s_route>",
        "<s_instructions>", "</s_instructions>"
    ]
    
    # Add special tokens to tokenizer
    num_added = processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    logger.info(f"Added {num_added} special structural tokens to tokenizer.")

    # Resize embedding layer to accommodate added tokens
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Configure generation parameters
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    task_token_id = processor.tokenizer.convert_tokens_to_ids(TASK_START_TOKEN)
    if task_token_id == processor.tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer did not register task token {TASK_START_TOKEN}")
    model.config.decoder_start_token_id = task_token_id
    model.config.eos_token_id = processor.tokenizer.convert_tokens_to_ids(EOS_TOKEN)

    # Create dataset
    train_dataset = PrescriptionDataset(
        dataset_path=args.dataset_path,
        processor=processor,
        max_length=args.max_length,
        split="train",
    )

    print_training_preview(train_dataset, processor)

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Enable half-precision only if on CUDA
        dataloader_num_workers=0 if os.name == 'nt' else 2, # Windows loader compatibility
        report_to="none" if not args.wandb else "wandb",
    )

    # Instantiate Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    logger.info("Starting training loop...")
    trainer.train()

    if args.verify_image:
        verification = run_inference_verification(
            model=model,
            processor=processor,
            image_path=Path(args.verify_image),
            device=device,
            max_length=args.max_length,
        )
        logger.info("Post-training inference verification: %s", json.dumps(verification, indent=2))
        if not has_structured_inference_output(verification):
            raise RuntimeError(
                "Inference verification returned empty structured output; refusing to package weights."
            )

    # Save final model and processor
    logger.info(f"Saving fine-tuned model and processor to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if args.package_zip:
        if not args.verify_image:
            raise ValueError("--package_zip requires --verify_image so weights are only packaged after inference passes")
        archive_base = str(Path(args.package_zip).with_suffix(""))
        archive_path = shutil.make_archive(archive_base, "zip", args.output_dir)
        logger.info("Packaged verified weights at %s", archive_path)

    logger.info("Training complete.")


def print_training_preview(train_dataset: PrescriptionDataset, processor: DonutProcessor) -> None:
    """Print records and one decoded target before any training begins."""
    print("\nValidated Donut training records:")
    for sample in iter_sample_summaries(train_dataset.samples, count=3):
        print(json.dumps(sample, indent=2, ensure_ascii=False))

    first_target = train_dataset.target_sequence(0)
    if "<s_medications></s_medications>" in first_target or re.search(r"<s_[^>]+>\s*</s_[^>]+>", first_target):
        raise ValueError(f"Decoded target contains empty tags: {first_target}")

    first_item = train_dataset[0]
    decoded = decoded_labels_to_text(first_item["labels"], processor.tokenizer, train_dataset.ignore_id)
    print("\nDecoded target sequence before training:")
    print(decoded)
    if "<s_iitcdip>" in decoded:
        raise ValueError("Decoded labels contain Donut base task token <s_iitcdip>; expected prescription task token")


def run_inference_verification(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    image_path: Path,
    device: str,
    max_length: int,
) -> dict[str, Any]:
    """Run one inference pass and parse the structured Donut output."""
    if torch is None:
        raise ImportError("Inference verification requires torch.")
    if Image is None:
        raise ImportError("Inference verification requires Pillow.")
    if not image_path.exists():
        raise FileNotFoundError(f"Verification image not found: {image_path}")

    model.eval()
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(
        TASK_START_TOKEN,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.convert_tokens_to_ids(EOS_TOKEN),
            use_cache=True,
            num_beams=1,
            do_sample=False,
        )

    sequence = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    sequence = sequence.replace(processor.tokenizer.pad_token or "", "").strip()
    if sequence.startswith(TASK_START_TOKEN):
        sequence = sequence[len(TASK_START_TOKEN):]
    sequence = sequence.replace(processor.tokenizer.eos_token or "", "").replace(EOS_TOKEN, "").strip()
    parsed = processor.token2json(sequence) if sequence else {}
    return {"raw_sequence": sequence, "parsed": parsed}


def has_structured_inference_output(result: dict[str, Any]) -> bool:
    """Return True only when inference produced useful structured fields."""
    parsed = result.get("parsed")
    if not isinstance(parsed, dict) or not parsed:
        return False
    if any(parsed.get(key) for key in ("doctor_name", "clinic_name", "patient_name", "patient_age", "prescription_date")):
        return True
    meds = parsed.get("medications")
    return isinstance(meds, list) and any(isinstance(med, dict) and med.get("drug_name") for med in meds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Donut on handwritten prescriptions.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Directory containing images and metadata.jsonl")
    parser.add_argument("--output_dir", type=str, default="./models/donut-rx", help="Directory to save fine-tuned model")
    parser.add_argument("--base_model", type=str, default="naver-clova-ix/donut-base", help="Hugging Face base model name")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max length of output token sequence")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--verify_image", type=str, default=None, help="Image to use for post-training inference verification")
    parser.add_argument("--package_zip", type=str, default=None, help="Optional zip path for verified weights")

    args = parser.parse_args()
    train(args)
