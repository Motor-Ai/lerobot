import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, Trainer
from datasets import load_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DATASET_NAME = 'yaak-ai/lerobot-driving-school'
MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEVICE = "cuda:2"

def load_trained_model(model_path="./outputs/smolvlm2-video-ft"):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def preprocess_dataset(dataset, processor):
    def preprocess(batch):
        inputs = processor(
            text=batch["text"],
            videos=batch["video"],
            return_tensors="pt",
            padding=True
        )
        return inputs

    return dataset.map(preprocess, batched=True)


def evaluate(model, processor, dataset):
    messages = [{"role": "user",
                 "content": [
                  {"type": "text", "text": "Follow the waypoints while adhering to traffic rules and regulations"},
                  {"type": "video", 
                   "path": "/home/user_lerobot/.cache/huggingface/lerobot/yaak-ai/lerobot-driving-school/videos/chunk-000/observation.images.front_center/episode_000000.mp4"}
                  ]
                }]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                          tokenize=True, return_dict=True, return_tensors="pt").to(DEVICE).to(model.dtype)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print(generated_texts[0])


def main():
    model, processor = load_trained_model(MODEL_NAME)
    dataset = load_dataset(DATASET_NAME)
    dataset = LeRobotDataset(DATASET_NAME)
    evaluate(model, processor, dataset)


if __name__ == "__main__":
    main()
