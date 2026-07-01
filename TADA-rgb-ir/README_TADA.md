# TADA RGB-IR SimMMDG 实验说明

本目录是针对 TADA 数据集新增的图像版 SimMMDG 实验入口。原仓库的 HAC/EPIC 代码面向视频、光流和音频；当前 TADA 数据是可见光与红外图像，因此这里保留 SimMMDG 的核心训练思想，并将模态后端替换为 RGB-IR 双图像编码器。

## 数据结构适配

脚本会自动兼容当前数据中的两种层级：

```text
晴天/train/可见光/1/*.jpg
晴天/train/红外/1/*.jpg
晴天/val/可见光/1/*.jpg
晴天/val/红外/1/*.jpg

黑天/train/1/可见光/*.jpg
黑天/train/1/红外/*.jpg
黑天/val/1/可见光/*.jpg
黑天/val/1/红外/*.jpg
```

`逆光`、`雾天`、`雨天` 与 `黑天` 一样，使用 `split/类别/模态` 结构。类别目录 `1` 到 `14` 会映射为标签 `0` 到 `13`。

默认配对策略是 `--pairing cycle`：

- 同名文件优先组成可见光-红外精确配对。
- 对未同名匹配的图片，在同一类别内按稳定排序轮转配对，尽量保留两个模态的样本。
- 每个 split 的精确配对数、轮转配对数、类别样本数会写入 `dataset_summary.json` 和 `dataset_preflight_summary.json`。

如果只希望使用严格同名样本，可以改为 `--pairing intersection`。如果要求两个模态文件名完全一致，可以使用 `--pairing strict`。

## 默认实验设计

默认设置对应你的实验要求：

- 源域：`晴天`
- 目标域：`黑天`、`逆光`、`雾天`、`雨天`
- 每个目标域独立实验
- 每个目标域重复 5 次，默认种子为 `0 1 2 3 4`
- 训练集：`晴天/train`
- 源域验证：`晴天/val`
- 目标测试集：目标天气的 `val`
- 模型：可见光 ResNet + 红外 ResNet，融合后分类
- SimMMDG 损失：分类损失、跨模态翻译损失、监督对比损失、共享/私有特征正交约束
- 测试指标：ACC、macro Precision、macro Recall、macro F1

默认是 source-only 训练，即只用 `晴天/train` 更新模型，并在目标天气 `val` 上测试。若你要做有标签目标域联合训练，可加 `--include_target_train`，脚本会把当前目标域的 `train` 也并入训练集。

## 运行命令

数据集路径不再写死。运行时可以用 `--data_root` 指定，也可以设置环境变量 `TADA_DATA_ROOT`。相对路径会按 `SimMMDG` 仓库根目录解析；输出目录默认保存在 `SimMMDG/runs/tada_simmmdg`。

先做数据扫描，不训练：

```bash
cd /path/to/SimMMDG
python TADA-rgb-ir/train_tada_simmmdg.py --data_root /path/to/TADA/Data --dry_run
```

也可以用环境变量：

```bash
cd /path/to/SimMMDG
export TADA_DATA_ROOT=/path/to/TADA/Data
python TADA-rgb-ir/train_tada_simmmdg.py --dry_run
```

运行完整 4 个目标域、每个 5 次重复实验：

```bash
cd /path/to/SimMMDG
python TADA-rgb-ir/train_tada_simmmdg.py \
  --data_root /path/to/TADA/Data \
  --source_domain 晴天 \
  --target_domains 黑天 逆光 雾天 雨天 \
  --repeats 5 \
  --epochs 30 \
  --batch_size 16 \
  --backbone resnet18 \
  --output_dir runs/tada_simmmdg
```

如需显式指定 5 次随机种子：

```bash
python TADA-rgb-ir/train_tada_simmmdg.py --data_root /path/to/TADA/Data --seeds 0 1 2 3 4
```

如果本机已有 ImageNet 预训练权重缓存，可加入：

```bash
--pretrained
```

服务器离线环境如果没有缓存，`--pretrained` 可能触发下载失败；不加该参数时会从零训练。当前数据量较小，建议优先使用可用的 ImageNet 预训练权重。

## 训练稳定性

早期版本使用 `-MSE(shared, private)` 作为特征拆分距离损失。该形式无下界，模型会通过无限放大 shared/private 特征范数来降低总 loss，表现为训练 loss 迅速变成巨大负数，随后分类结果退化到固定类别。当前默认 `--split_loss orthogonal` 会先归一化 shared/private 特征，再最小化二者 cosine 相似度平方，目标有界且更稳定。

如果需要复现实验问题，可显式指定：

```bash
--split_loss negative_mse
```

不建议在正式实验中使用该选项。

## 输出文件

每次运行会创建：

```text
SimMMDG/runs/tada_simmmdg/weather_rgb_ir_YYYYmmdd_HHMMSS/
```

主要文件：

- `config.json`：完整参数配置
- `dataset_preflight_summary.json`：所有源域/目标域 train/val 的数据扫描摘要
- `{目标域}/repeat_XX_seed_Y/run.log`：单次重复的训练日志
- `{目标域}/repeat_XX_seed_Y/epoch_metrics.csv`：每个 epoch 的 train、source_val、target_test 指标
- `{目标域}/repeat_XX_seed_Y/dataset_summary.json`：该次重复实际使用的数据统计
- `{目标域}/repeat_XX_seed_Y/repeat_summary.json`：该次重复测试集最高 ACC、Precision、Recall、F1 及对应 epoch
- `all_repeats_summary.csv`：所有目标域与重复轮次的汇总
- `aggregate_summary.csv`：每个目标域 5 次重复后的均值和标准差

默认按目标测试集 `F1` 选择 `best epoch`，同时独立记录 ACC、Precision、Recall、F1 各自的最高值。可以用 `--best_metric acc` 改成按 ACC 保存最佳模型。
