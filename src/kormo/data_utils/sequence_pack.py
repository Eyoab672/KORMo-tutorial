from itertools import chain

def _pack_dataset(examples, seq_len):
    flat = list(chain.from_iterable(examples["input_ids"]))
    n_full  = len(flat) // seq_len
    chunks  = [flat[i*seq_len:(i+1)*seq_len] for i in range(n_full)]

    return {"input_ids": chunks}

def pack_dataset(ds, seq_len):
    return ds.map(
        _pack_dataset, 
        batched=True, 
        batch_size=100_000, 
        remove_columns=ds.column_names, 
        num_proc=128,
        fn_kwargs={'seq_len': seq_len}
    )