```
ML_Project/
├── configs/
│   └── experiment_configs.yaml          # TODO: consider to delete
├── data/
│   ├── sst2_dataset/
│   ├── ag_news_dataset/                
│   ├── imdb_dataset/                    
│   └── tweet_eval_dataset/              
├── src/
│   ├── LoRa/
│   │   └── components/
│   │       ├── data_loader/
│   │       │   ├── base.py
│   │       │   ├── huggingface_data_loader.py  
│   │       │   └── enhanced_data_loader.py     
│   │       ├── huggingface_models/
│   │       │   ├── base.py
│   │       │   └── huggingface_model_loader.py
│   │       ├── peft/
│   │       │   ├── base.py
│   │       │   ├── adapter_builder.py
│   │       │   ├── lora_builder.py
│   │       │   ├── hybrid_builder.py
│   │       │   ├── peft_factory.py             # Unified entry point
│   │       │   └── peft_model_builder.py       
│   │       └── trainer/
│   │           ├── base.py
│   │           └── experiment_trainer.py       
│   └── main/
│       ├── improved_experiment_runner.py       # Uses factory pattern
│       ├── batch_experiment_runner.py          # Runs all experiments 
│       ├── main_experiment_runner.py           # KEEP for backward compatibility
│       └── analyse_results.py
└── results/
    └── [experiment outputs]
```
