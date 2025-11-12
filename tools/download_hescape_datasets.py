# tools/download_hescape_datasets.py
from datasets import load_dataset
from pathlib import Path

PANELS = [
    "human-5k-panel",
    "human-breast-panel",
    "human-colon-panel",
    "human-immuno-oncology-panel",
    "human-lung-healthy-panel",
    "human-multi-tissue-panel",
]
SPLITS = ["train", "validation", "test"]

def main(cache_dir="~/.cache/hescape_hf"):
    cache_dir = str(Path(cache_dir).expanduser())
    for name in PANELS:
        for split in SPLITS:
            print(f"==> downloading {name} [{split}]")
            _ = load_dataset(
                "Peng-AI/hescape-pyarrow",
                name=name,
                split=split,
                streaming=False,
                cache_dir=cache_dir,
                num_proc=4,
            )
    print("Done.")

if __name__ == "__main__":
    main()
