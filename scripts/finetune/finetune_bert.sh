 #!/bin/bash
python -m treccast.reranker.train.bert_reranker_train --gpus 1 2 6 7 \
        --accelerator dp  --warmup_steps 1000 --max_epochs 20 --save_top_k=3 \
        --lr 3e-05 --val_patience 5 --val_metric val_RetrievalNormalizedDCG \
        --dropout 0.3