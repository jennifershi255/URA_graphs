import pandas as pd
df = pd.read_csv('records_final.csv')
df.to_csv('records_final.csv', index=False)
print(df.columns.tolist())
print(df.shape)
print(df.head(3))
print(df['finetuned_dataset'].value_counts())

# normalize dataset names
name_map = {
    'emotion': 'dair-ai/emotion',
    'imdb': 'stanfordnlp/imdb',
    'go_emotions': 'google-research-datasets/go_emotions',
    '**ViSpamReviews**': 'ViSpamReviews',
    '**VSMEC**': 'visolex/VSMEC',
}
df['finetuned_dataset'] = df['finetuned_dataset'].replace(name_map)

# drop datasets with no embeddings
datasets_with_embeddings = [
    'contemmcm/hate-speech-and-offensive-language',
    'dair-ai/emotion',
    'enguard/multi-lingual-prompt-moderation',
    'google/boolq',
    'google-research-datasets/go_emotions',
    'stanfordnlp/imdb',
    'Tevatron/msmarco-passage',
    'ToxicityPrompts/PolyGuardMix',
    'trl-internal-testing/tldr-preference-sft-trl-style',
    'visolex/vihsd',
]
df = df[df['finetuned_dataset'].isin(datasets_with_embeddings)]

print(df['finetuned_dataset'].value_counts())
print(df.shape)

df.to_csv('records_final.csv', index=False)
print("Saved records_final.csv with", len(df), "rows")