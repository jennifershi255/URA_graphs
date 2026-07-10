from huggingface_hub import hf_hub_download

model2family_path = hf_hub_download("luisrui/ModelLens-corpus-v2", "model2family.json", repo_type="dataset")
model_profile_path = hf_hub_download("luisrui/ModelLens-corpus-v2", "model_profile.json", repo_type="dataset")
print(model2family_path)
print(model_profile_path)