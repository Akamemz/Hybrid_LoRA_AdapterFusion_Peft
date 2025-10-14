#!/usr/bin/env python
"""
run_sst2.py — Minimal SST-2 runner for LoRA / Adapters / AdapterFusion / Hybrid (LoRA+Fusion)
Uses adapter-transformers so everything coexists cleanly.

Examples:
    LoRA:      python run_sst2.py --method lora --rank 8
    Adapter:   python run_sst2.py --method adapter --reduction 16
    Fusion:    python run_sst2.py --method fusion --fusion_sources /path/agnews_adapter /path/imdb_adapter
    Hybrid:    python run_sst2.py --method hybrid --rank 8 --fusion_sources /path/agnews_adapter /path/imdb_adapter
"""
import argparse, math, os, random, sys, json, time
from typing import List, Optional

import torch
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# Adapter-Transformers APIs
from transformers.adapters.composition import Fuse, Stack
from transformers.adapters import AdapterConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default="bert-base-uncased")
    p.add_argument("--method", type=str, required=True,
                   choices=["lora", "adapter", "fusion", "hybrid"])
    p.add_argument("--rank", type=int, default=8, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--reduction", type=int, default=16, help="Adapter bottleneck (Pfeiffer)")
    p.add_argument("--fusion_sources", type=str, nargs="*", default=[],
                   help="Paths or hub IDs of source adapters to fuse (>=2)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_train_samples", type=int, default=-1, help="Few-shot cap; -1 uses full data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--use_fp16", action="store_true")
    return p.parse_args()


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, trainable/total


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = f"sst2_{args.method}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Data
    ds = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)

    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True)

    ds = ds.map(tok, batched=True)
    if args.num_train_samples and args.num_train_samples > 0:
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(min(args.num_train_samples, len(ds["train"]))))
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 2) Model
    config = AutoConfig.from_pretrained(args.backbone, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(args.backbone, config=config)

    # 3) Configure adapters/LoRA
    # We keep the classification head trainable together with the chosen adapters.
    if args.method == "adapter":
        ad_cfg = AdapterConfig.load("pfeiffer", reduction_factor=args.reduction, non_linearity="relu")
        model.add_adapter("sst2_adapter", config=ad_cfg)
        model.train_adapter("sst2_adapter")
        model.set_active_adapters("sst2_adapter")

    elif args.method == "lora":
        lora_cfg = AdapterConfig.load("lora", r=args.rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
                                      target_modules=["query", "value"])  # typical q,v
        model.add_adapter("sst2_lora", config=lora_cfg)
        model.train_adapter("sst2_lora")
        model.set_active_adapters("sst2_lora")

    elif args.method in ["fusion", "hybrid"]:
        assert len(args.fusion_sources) >= 2, "--fusion_sources needs >=2 adapters (paths or hub IDs)."
        # Load source adapters (frozen for target), name them src_0, src_1, ...
        src_names = []
        for i, src in enumerate(args.fusion_sources):
            name = f"src{i}"
            # source can be local path or hub ID (adapter-transformers handles both)
            model.load_adapter(src, load_as=name)
            src_names.append(name)

        # Add and activate fusion of sources
        comp_fuse = Fuse(*src_names)
        model.add_adapter_fusion(comp_fuse, "sst2_fusion")
        model.train_adapter_fusion("sst2_fusion")

        if args.method == "fusion":
            model.set_active_adapters(comp_fuse)  # only fuse sources
        else:
            # HYBRID: add LoRA on top and stack: [LoRA] -> [FUSE(sources)]
            lora_cfg = AdapterConfig.load("lora", r=args.rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
                                          target_modules=["query", "value"])
            model.add_adapter("sst2_lora", config=lora_cfg)
            # Train both: fusion & LoRA
            model.train_adapter("sst2_lora")
            comp = Stack("sst2_lora", comp_fuse)
            model.set_active_adapters(comp)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # 4) Report trainable params
    trainable, total, frac = count_trainable_params(model)
    print(f"[PARAMS] trainable={trainable:,}  total={total:,}  frac={frac:.4%}")

    # 5) Trainer
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"accuracy": metric.compute(predictions=preds, references=labels)["accuracy"]}

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=args.use_fp16,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # 6) Train + Eval
    trainer.train()
    val_metrics = trainer.evaluate(ds["validation"])
    test_metrics = trainer.evaluate(ds["test"])
    print(f"[VAL]  {val_metrics}")
    print(f"[TEST] {test_metrics}")

    # 7) Save artifacts
    if args.method == "adapter":
        model.save_adapter(os.path.join(out_dir, "sst2_adapter"), "sst2_adapter")
    elif args.method == "lora":
        model.save_adapter(os.path.join(out_dir, "sst2_lora"), "sst2_lora")
    elif args.method == "fusion":
        model.save_adapter_fusion(os.path.join(out_dir, "sst2_fusion"), "sst2_fusion")
    elif args.method == "hybrid":
        model.save_adapter(os.path.join(out_dir, "sst2_lora"), "sst2_lora")
        model.save_adapter_fusion(os.path.join(out_dir, "sst2_fusion"), "sst2_fusion")

    # 8) Persist run summary
    summary = {
        "args": vars(args),
        "val": val_metrics,
        "test": test_metrics,
        "trainable_params": int(trainable),
        "total_params": int(total),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
