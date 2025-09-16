import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from peft import PeftModel

# -----------------------------
# Config
# -----------------------------
DATASET_NAME = "yaak-ai/lerobot-driving-school"
BASE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
ADAPTER_PATH = "./outputs/bs_8_SmolVLM2-500M-Video-Instruct-driving_last"
DEVICE = "cuda:2"
USE_FINETUNED = True  # switch between base and finetuned
# -----------------------------


def load_trained_model(use_finetuned: bool = True):
    # Load base model
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )

    # Attach LoRA adapters if finetuned flag is set
    if use_finetuned:
        if not os.path.isdir(ADAPTER_PATH):
            raise FileNotFoundError(f"Adapter path not found: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # Always use processor from base model
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    return model, processor


def preprocess_dataset(dataset, processor):
    def preprocess(batch):
        inputs = processor(
            text=batch["text"],
            videos=batch["video"],
            return_tensors="pt",
            padding=True,
        )
        return inputs

    return dataset.map(preprocess, batched=True)


def evaluate(model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Follow the waypoints while adhering to traffic rules and regulations"},
                {
                    "type": "video",
                    "path": "/home/user_lerobot/.cache/huggingface/lerobot/yaak-ai/lerobot-driving-school/videos/chunk-000/observation.images.front_center/episode_000003.mp4",
                },
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE).to(model.dtype)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print("\n--- Model Output ---")
    print(generated_texts[0])


def main():
    model, processor = load_trained_model(use_finetuned=USE_FINETUNED)

    evaluate(model, processor)


if __name__ == "__main__":
    main()
