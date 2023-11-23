import os
import pandas as pd
from datasets import load_dataset


def prepare_default_dataset(output_path="oasst1"):
    ds = load_dataset("OpenAssistant/oasst1")
    train = ds["train"].to_pandas()
    val = ds["validation"].to_pandas()

    df = pd.concat([train, val], axis=0).reset_index(drop=True)

    df_assistant = df[(df.role == "assistant")].copy()
    df_prompter = df[(df.role == "prompter")].copy()
    df_prompter = df_prompter.set_index("message_id")
    df_assistant["output"] = df_assistant["text"].values

    inputs = []
    parent_ids = []
    for _, row in df_assistant.iterrows():
        input = df_prompter.loc[row.parent_id]
        inputs.append(input.text)
        parent_ids.append(input.parent_id)

    df_assistant["instruction"] = inputs
    df_assistant["parent_id"] = parent_ids

    df_assistant = df_assistant[
        ["instruction", "output", "message_id", "parent_id", "lang", "rank"]
    ].rename(columns={"message_id": "id"})
    
    path = os.path.join(output_path, "train_full.jsonl")
    df_assistant.to_json(path, orient="records", lines=True)
    dataset = load_dataset("json", data_files=path)
    dataset.push_to_hub("studio-ousia/oasst1-instruction")

    return df_assistant[(df_assistant["rank"] == 0.0) & (df_assistant["lang"] == "en")]


def prepare_default_dataset_ja(output_path="oasst1-89k-ja"):
    ds = load_dataset("kunishou/oasst1-89k-ja")
    df = ds["train"].to_pandas()

    df_assistant = df[(df.role == "assistant")].copy()
    df_prompter = df[(df.role == "prompter")].copy()
    df_prompter = df_prompter.set_index("message_id")
    df_assistant["output"] = df_assistant["text_ja"].values

    inputs = []
    parent_ids = []
    for _, row in df_assistant.iterrows():
        input = df_prompter.loc[row.parent_id]
        inputs.append(input.text_ja)
        parent_ids.append(input.parent_id)

    df_assistant["instruction"] = inputs
    df_assistant["parent_id"] = parent_ids

    df_assistant = df_assistant[
        ["instruction", "output", "message_id", "parent_id", "lang"]
    ].rename(columns={"message_id": "id"})
    
    path = os.path.join(output_path, "train_full.jsonl")
    df_assistant.to_json(path, orient="records", lines=True)
    dataset = load_dataset("json", data_files=path)
    dataset.push_to_hub("studio-ousia/oasst1-89k-ja-instruction")

    return df_assistant


if __name__ == "__main__":
    prepare_default_dataset()
