# Leash Bio BELKA

## Kaggle Competition

[https://www.kaggle.com/competitions/leash-BELKA/overview](https://www.kaggle.com/competitions/leash-BELKA/overview)

주어진 약물 분자(Molecule)가 표적 단백질(Protein)과 합성될 가능성을 예측, 이진 분류하는 문제. 화학 분야에서는 Drug-Target Interaction, DTI라고 부른다.

## Installation

Pytorch와 DGL은 GNN을 실험하기 위한 것으로 필수 설치가 아니다.

    ## Pytorch(Optional)
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

    ## DGL(Optional)
    pip install dgl -f https://data.dgl.ai/wheels/repo.html

    ## Others
    pip install -r requirements.txt

## Training

### 1.`config.yaml` 수정

    debug: false
    prepare: false

    data_dir: "/home/pervinco/Datasets/leash-bio"
    save_dir: "./runs"
    ckpt_dir: "./runs/2024-06-15-15-28-33" # "/home/pervinco/Leash-Bio/runs/2024-06-12-14-48-33"

    train_parquet: "/home/pervinco/Datasets/leash-bio/train_split.parquet"
    valid_parquet: "/home/pervinco/Datasets/leash-bio/valid_split.parquet"
    test_parquet: "/home/pervinco/Datasets/leash-bio/test.parquet"

    n_iter: 20
    limit: 200000
    radius: 2
    vec_dim: 2048

    num_boost: 10000
    learning_rate: 0.005
    feature_frac: 1.0
    num_leaves: 62
    max_depth: 50
    l1_lambda: 0.0
    l2_lambda: 0.2
    drop_prob: 0.3

    predict_top_k: 10
    feature_top_n: 20

    target_proteins:
    - HSA
    - sEH
    - BRD4

    uniprot_dicts:
    sEH: "P34913"
    BRD4: "O60885"
    HSA: "P02768"

### 2.학습

    python train.py

### 3.테스트(제출 파일 생성)

`config.yaml`에서 ckpt_dir을 테스트하려는 경로로 설정. ex)`"./Leash-Bio/runs/2024-06-12-14-48-33"`

    python test.py
