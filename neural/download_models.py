"""Download HuggingFace models to cache."""

import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Download models to HuggingFace cache")
    parser.add_argument("models", nargs="+", help="Model names to download")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    for model_name in args.models:
        print(f"Downloading {model_name}...")
        try:
            AutoTokenizer.from_pretrained(model_name, token=hf_token)
            AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
            print(f"✓ {model_name}")
        except Exception as e:
            print(f"✗ {model_name}: {e}")
            raise


if __name__ == "__main__":
    main()
