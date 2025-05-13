class LocalModel:
    dict={
        "facebook/opt-125m": "/lamport/shared/yujiema/opt-125m/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/",
        "facebook/opt-350m": "/lamport/shared/yujiema/opt-350m/models--facebook--opt-350m/snapshots/cb32f77e905cccbca1d970436fb0f5e6b58ee3c5/",
        "facebook/opt-1.3b": "/lamport/shared/yujiema/opt-1.3b/models--facebook--opt-1.3b/snapshots/8c7b10754972749675d22364c25c428b29face51/",
        "facebook/opt-2.7b": "/lamport/shared/yujiema/opt-2.7b/models--facebook--opt-2.7b/snapshots/905a4b602cda5c501f1b3a2650a4152680238254/",
        "facebook/opt-6.7b": "/lamport/shared/yujiema/opt-6.7b/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0/",
        "facebook/opt-13b": "/lamport/shared/yujiema/opt-13b/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/",
        "facebook/opt-30b": "/lamport/shared/yujiema/opt-30b/models--facebook--opt-30b/snapshots/ceea0a90ac0f6fae7c2c34bcb40477438c152546/",
        "meta/llama2-7b": "/lamport/shared/yujiema/llama-2-7b-hf/models--meta-llama--llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9/",
        "openbmb/MiniCPM-2B-128k": "/lamport/shared/yujiema/minicpm-2.4b/snapshots/6011acb51acf3e8c0cb10c428d7064dc39431720/"
    }

class LocalModelLoad(LocalModel):
    @classmethod
    def from_pretrained(cls, model_info, **kwargs):
        from transformers import AutoModelForCausalLM
        model_path=cls.dict.get(model_info, "")
        if model_path=="":
            return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_info, **kwargs)
        else:
            return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, **kwargs)

class LocalModelConfig(LocalModel):
    @classmethod
    def from_pretrained(cls, model_info, **kwargs):
        from transformers import AutoConfig
        model_path=cls.dict.get(model_info, "")
        if model_path=="":
            return AutoConfig.from_pretrained(pretrained_model_name_or_path=model_info, **kwargs)
        else:
            return AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path, **kwargs)

class LocalModelTokenizer(LocalModel):
    @classmethod
    def from_pretrained(cls, model_info, **kwargs):
        from transformers import AutoTokenizer
        model_path=cls.dict.get(model_info, "")
        if model_path=="":
            return AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_info, **kwargs)
        else:
            return AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, **kwargs)

class LocalData:
    dict={
        "wikitext": {
            "wikitext-2-raw-v1": {
                "train": "/lamport/shared/yujiema/wikitext/wikitext-2-raw-v1-train",
                "validation": "/lamport/shared/yujiema/wikitext/wikitext-2-raw-v1-validation",
                "test": "/lamport/shared/yujiema/wikitext/wikitext-2-raw-v1-test",
            },
        },
        "ptb_text_only": {
            "penn_treebank": {
                "train": "/lamport/shared/yujiema/ptb_text_only/penn_treebank-train",
                "validation": "/lamport/shared/yujiema/ptb_text_only/penn_treebank-validation",
                "test": "/lamport/shared/yujiema/ptb_text_only/penn_treebank-test",
            },
        },
        "allenai/c4": {
            "allenai--c4": {
                "train": "/lamport/shared/yujiema/allenai/c4/allenai--c4-train",
                "validation": "/lamport/shared/yujiema/allenai/c4/allenai--c4-validation",
                "test": "/lamport/shared/yujiema/allenai/c4/allenai--c4-test",
            },
        },
    }

class LocalDataLoad(LocalData):
    @classmethod
    def load_dataset(cls, data, dataset, **kwargs):
        from datasets import load_dataset, load_from_disk
        data_path=cls.dict.get(data, "")
        data_path=data_path.get(dataset, "") if data_path!="" else ""
        data_path=data_path.get(kwargs["split"], "") if data_path!="" else ""
        if data_path=="":
            return load_dataset(data, dataset, **kwargs)
        else:
            return load_from_disk(data_path)
