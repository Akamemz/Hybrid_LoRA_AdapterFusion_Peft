```
ML_Project/
├── configs/
│   └── experiment_configs.yaml          # NEW: Centralized config
├── data/
│   ├── sst2_dataset/
│   ├── ag_news_dataset/                 # TO ADD
│   ├── imdb_dataset/                    # TO ADD
│   └── tweet_eval_dataset/              # TO ADD
├── src/
│   ├── LoRa/
│   │   └── components/
│   │       ├── data_loader/
│   │       │   ├── base.py
│   │       │   ├── huggingface_data_loader.py  # KEEP for backward compatability
│   │       │   └── enhanced_data_loader.py     # NEW: Multi-dataset support
│   │       ├── huggingface_models/
│   │       │   ├── base.py
│   │       │   └── huggingface_model_loader.py
│   │       ├── peft/
│   │       │   ├── base.py
│   │       │   ├── adapter_builder.py
│   │       │   ├── lora_builder.py
│   │       │   ├── hybrid_builder.py
│   │       │   ├── peft_factory.py             # NEW: Unified entry point
│   │       │   └── peft_model_builder.py       # KEEP for backward compatability
│   │       └── trainer/
│   │           ├── base.py
│   │           └── experiment_trainer.py       # FIX: Remove BaseTrainer import
│   └── main/
│       ├── improved_experiment_runner.py       # NEW: Uses factory pattern
│       ├── batch_experiment_runner.py          # NEW: Runs all experiments
│       ├── main_experiment_runner.py           # KEEP for backward compatability
│       ├── experiment_orchestrator.py          # UPDATE or deprecate (Unsure)
│       └── analyse_results.py
└── results/
    └── [experiment outputs]
```
