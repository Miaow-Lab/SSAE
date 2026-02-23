# Classifier Experiment

本目录包含论文中第一个实验（classifier）的全部代码，流程如下：

1. 使用 `classifier_data.py` 生成训练数据：
   - `*.npz`：输入特征与标签（`latents/hints/step_length/begin_token_id/correctness/ids`）
   - `*_judge.jsonl`：用于调用 LLM API 的 prompt（主要用于 Correctness/Logicality）
2. 调用外部 LLM API（仓库内不包含 API 调用代码）得到回复
3. 使用 `combine_label.py` 将 API 回复中的 `Correctness/Logicality` 标签合并回 npz
4. 使用 `classifier_train.py` 训练不同标签的 classifier
5. `step length / first token ppl` 可直接看训练 loss；`correctness/logicality` 用 `correctness_eval.py` 评估
6. 四个标签的统计 baseline 使用 `statistic_baseline.py`

所有入口脚本统一使用同一个总配置文件：`configs/classifier.yaml`，
每个脚本只读取自己的 section（如 `classifier_data`、`combine_label`、`classifier_train` 等）。

## 1) 生成 classifier 数据

```bash
bash scripts/run_classifier_data.sh configs/classifier.yaml
```

按需覆盖参数（仅覆盖 `classifier_data` section 的键）：

```bash
bash scripts/run_classifier_data.sh configs/classifier.yaml \
  --set checkpoint_name=ckpt_xxx.pt \
  --set input_file=data/gsm8k_385K_valid.json \
  --set output_file=classifier_data/gsm8k_385K_valid_classifier_data
```

## 2) 调用 LLM API（外部）

- 输入：上一步生成的 `*_judge.jsonl`
- 输出：需要至少包含字段：`id` 与 `response`（或你自定义的响应字段）

## 3) 合并 Correctness/Logicality 标签

```bash
python classifier/combine_label.py --config configs/classifier.yaml
```

如果 API 响应字段不是 `response`，可用：

```bash
python classifier/combine_label.py --config configs/classifier.yaml --set response_key=content
```

## 4) 训练 classifier

```bash
bash scripts/run_classifier_train.sh configs/classifier.yaml
```

按需覆盖参数（仅覆盖 `classifier_train` section 的键）：

```bash
PROC=4 bash scripts/run_classifier_train.sh configs/classifier.yaml \
  --set task=correctness \
  --set inputs=hints \
  --set train_file=classifier_data/train_labeled.npz \
  --set val_file=classifier_data/valid_labeled.npz
```

## 5) 评估 Correctness/Logicality

```bash
bash scripts/run_correctness_eval.sh configs/classifier.yaml
```

按需覆盖参数（仅覆盖 `correctness_eval` section 的键）：

```bash
bash scripts/run_correctness_eval.sh configs/classifier.yaml \
  --set ckpt=out_classifier/exp_correctness_hints/ckpt_xxx.pt \
  --set data_file=classifier_data/valid_labeled.npz \
  --set task=correctness \
  --set inputs=hints
```

## 6) 统计 baseline

```bash
python classifier/statistic_baseline.py --config configs/classifier.yaml
```

或覆盖：

```bash
python classifier/statistic_baseline.py --config configs/classifier.yaml \
  --set train_npz=classifier_data/train_labeled.npz \
  --set val_npz=classifier_data/valid_labeled.npz \
  --set show_logicality=true
```

输出包括：
- `first token` 的 entropy / perplexity baseline
- `step length` 的 mean/std baseline
- `correctness`（以及可选 `logicality`）的正例比例 baseline
