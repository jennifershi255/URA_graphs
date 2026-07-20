import pandas as pd

df = pd.read_csv('records_dense_with_base_models.csv')

# drop rows with missing base_model
df = df.dropna(subset=['base_model'])

# rename base_model -> model, drop the original model column
df = df.drop(columns=['model'])
df = df.rename(columns={'base_model': 'model'})
df = df.rename(columns={'num_epochs': 'num_train_epochs'})
# add missing columns TransferGraph expects with null values
for col in ['model_name', 'dataset_name', 'lr_scheduler_type', 
            'gradient_accumulation_steps', 'seed', 'dataset_path', 
            'train_runtime', 'push_to_hub', 'hub_model_id', 
            'push_to_hub_organization', 'hub_token', 'peft_method',
            'lora_attention_dimension', 'lora_alpha', 'lora_dropout', 'lora_bias']:
    if col not in df.columns:
        df[col] = None

# reorder columns to match TransferGraph
cols = ['model', 'finetuned_dataset', 'model_name', 'dataset_name', 'task_type',
        'batch_size', 'lr_scheduler_type', 'learning_rate', 'gradient_accumulation_steps',
        'num_train_epochs', 'seed', 'dataset_path', 'train_runtime', 'eval_accuracy',
        'push_to_hub', 'hub_model_id', 'push_to_hub_organization', 'hub_token',
        'peft_method', 'lora_attention_dimension', 'lora_alpha', 'lora_dropout', 'lora_bias']

df = df[cols]
df.to_csv('records_final.csv', index=True)
print(df.shape)
print(df.head())