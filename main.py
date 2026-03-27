import logging
import torch
import pytorch_lightning as pl
from src.data.data import split_by_task, load_aste_file, Task, to_generative_format, filter_tasks, _encode_target, _decode_target
from src.eval.eval import parse_output
from src.model.model import T5ABSAModel

TRAIN_FILE = "downloads/ABSADatasets/datasets/aste_datasets/400.SemEval/402.Restaurant14/train.txt"
VAL_FILE   = "downloads/ABSADatasets/datasets/aste_datasets/400.SemEval/402.Restaurant14/dev.txt"

TASKS_PARTITION = {
    (Task.ASPECT,):                               0.3,
    (Task.POLARITY,):                             0.3,
    (Task.SENTIMENT,):                            0.3,
    (Task.ASPECT, Task.POLARITY, Task.SENTIMENT): 0.1,
}

def test_bracket_format():
    triplets = [
        {"aspect": "food", "sentiment": "outstanding", "polarity": "positive"},
        {"aspect": "service", "sentiment": "slow", "polarity": "negative"},
    ]
    sentence = "The food was outstanding but the service was slow."

    # 1. encode -> decode round-trip
    encoded = _encode_target(triplets)
    decoded = _decode_target(encoded, list(triplets[0].keys()))
    assert decoded == triplets, f"Round-trip failed: {decoded}"

    # 2. to_generative_format produces bracket target
    ex = to_generative_format(sentence, triplets)
    assert ex["target"] == encoded, f"to_generative_format target mismatch: {ex['target']}"
    assert "[" in ex["target"] and "]" in ex["target"]

    # 3. filter_tasks keeps only requested keys
    filtered = filter_tasks(ex, [Task.ASPECT, Task.POLARITY])
    keys = filtered["_keys"]
    assert keys == ["aspect", "polarity"], f"Wrong keys after filter: {keys}"
    parsed = parse_output(filtered["target"], keys)
    assert parsed == [{"aspect": "food", "polarity": "positive"}, {"aspect": "service", "polarity": "negative"}], \
        f"filter_tasks parse mismatch: {parsed}"

    # 4. parse_output on a well-formed prediction
    pred_raw = "[food, outstanding, positive] [service, slow, negative]"
    parsed_pred = parse_output(pred_raw, ["aspect", "sentiment", "polarity"])
    assert parsed_pred == triplets, f"parse_output mismatch: {parsed_pred}"

    # 5. parse_output on malformed input returns empty
    assert parse_output("not a bracket format", ["aspect"]) == []

    print("All bracket format tests passed.")
    print(f"  encoded target : {encoded}")
    print(f"  filtered target: {filtered['target']}")


def main():
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(42, workers=True)
    test_bracket_format()
    train_partitions = split_by_task(TRAIN_FILE, TASKS_PARTITION, shuffle_tasks=True)
    train_examples = [ex for part in train_partitions.values() for ex in part]
    val_examples   = load_aste_file(VAL_FILE)

    print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    model = T5ABSAModel(
        model_name="google/flan-t5-base",
        learning_rate=3e-4,
        max_length=256,
        batch_size=8,
        train_examples=train_examples,
        val_examples=val_examples,
        eval_keys=[
            ["aspect", "sentiment", "polarity"],  # full triplet
            ["aspect"],
            ["sentiment"],
            ["polarity"],
        ],
    )

    trainer = pl.Trainer(
        max_epochs=3,
        log_every_n_steps=1,
        limit_train_batches=1.0,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        enable_model_summary=True,
        precision="bf16-mixed",
        accumulate_grad_batches=4,
    )
    trainer.fit(model)

if __name__ == "__main__":
    main()
