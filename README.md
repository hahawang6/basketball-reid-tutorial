# 球眸智枢：球员重识别训练教程

使用篮球球员裁剪图片学习身份特征，通过 Query 在 Gallery 中检索相似球员。本项目包含 SwinReID 网络、图片处理、损失函数、训练与断点恢复、检索评价及图片检索代码。

操作顺序为：环境配置 → 图片准备 → 身份划分 → 预训练模型微调 → 独立测试 → 图片检索。

训练、断点恢复和完整数据集评价尚未用真实数据完成端到端实测。已有检查覆盖网络前向接口、身份编号、采样约束、检索指标、划分规则和随机状态保存恢复；小图库真实权重检索曾完成计算，但不构成数据集精度结果。

## 目录

```text
basketball-reid-tutorial/
  README.md
  LICENSE
  THIRD_PARTY_NOTICES.md
  requirements.txt
  training.example.json
  retrieve.py
  src/
    model.py             SwinReID 网络与特征输出
    dataset.py           图片加载、增强、身份编号与采样
    loss.py              ArcFace、条带分类、Circle 和 Triplet 损失
    evaluate.py          余弦距离、CMC 与 mAP
    data_splits.py       身份隔离与图片 SHA-256 校验
    training_config.py   配置、设备和输出目录约束
    training_state.py    随机状态与检查点写入
    checkpoints.py       受限读取检查点
  tools/
    prepare_reid_split.py
  training/
    train_reid.py
    evaluate_reid.py
  tests/
    test_reid.py
    test_data_splits.py
    test_resume.py
```

## 环境配置

参考环境为 Linux、Python 3.12、CPU。以下命令在包含本 README 的项目根目录执行；

先安装 Python 3.12，再建立独立环境：

```bash
python3.12 --version
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

`pip check` 应没有依赖冲突。后续始终使用 `.venv/bin/python`，避免装入其他解释器。CPU 版不能通过改变命令中的设备参数变成 GPU 版；CUDA 环境需要另按 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/)准备。

## 数据准备

下载 [BallShow-ReID 图片数据](https://pan.baidu.com/s/1VlMJSKfvhsynJrrSuO9nqQ?pwd=rqjz)（提取码：`rqjz`），数据获取入口来源见第三方引用。

解压后组织为：

```text
data/original/BallShow-ReID/
  bounding_box_train/
  query/
  bounding_box_test/
```

`bounding_box_train` 是身份训练图片，`query` 是查询图片，`bounding_box_test` 是候选图库。图片已经是球员裁剪图，不需要先运行目标检测。

例如 `0002_c1s1_000016_00.jpg` 的身份是 `2`。训练将身份重新编号为连续分类标签；Query 和 Gallery 始终保留原身份编号。

程序在读取时完成 224×224 尺寸变换、RGB 归一化和训练增强，不生成另一份预处理图片。验证、测试不应用训练随机增强。图片路径和名称需要符合 `src/dataset.py` 的加载规则；无法读取的图片会报错。

## 配置和数据划分

第一次使用时复制配置，不覆盖已有实验配置：

```bash
cp -n training.example.json training.local.json
```

`training.local.json` 中的路径相对于该配置文件所在目录解析。默认参数如下：

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `data_root` | `data/original/BallShow-ReID` | 包含三个原始图片目录 |
| `split_file` | `data/splits/reid.json` | 固定划分清单 |
| `output_dir` | `runs/reid-01` | 本次实验产物 |
| `batch_size` | 8 | 每批图片数 |
| `num_instances` | 4 | 每个身份的采样数 |
| `epochs` | 80 | 总训练轮数 |
| `learning_rate` | 0.0003 | 分类头与损失参数的基础学习率 |
| `eval_every` | 5 | 每多少轮进行验证并选择权重 |
| `eval_batch_size` | 8 | 验证与特征提取的批量 |
| `seed` | 42 | 划分和训练随机种子 |

`batch_size` 必须能被 `num_instances` 整除，一批至少两个身份，每个身份至少两张图片。显存或内存不足时可以改为 `batch_size=4`、`num_instances=2`，并使用新的实验目录。

执行：

```bash
.venv/bin/python -B tools/prepare_reid_split.py --config training.local.json
```

输出 `data/splits/reid.json`，并打印各分区的图片数量。此操作不移动或改写原图；已有清单不覆盖。

划分规则：

- 在原始训练身份中按固定种子留出约 20% 作验证身份，其余身份用于训练。
- 验证身份需要至少两张内容不同的图片：一张作验证 Query，其余作验证 Gallery。
- 原始 `query` 和 `bounding_box_test` 只用于最终测试，不参与训练中的权重选择。
- 训练、验证、测试的身份必须隔离；跨分区完全相同的图片内容会被拒绝。

清单保存相对路径和 SHA-256。训练和评价会核对图片内容；图片或划分改变后不能继续复用该实验。

## 模型和预训练权重

`src/model.py` 的 `SwinReID` 使用 timm 的 `swin_base_patch4_window7_224` 骨干；全局分支使用 GeM 池化，局部分支使用六条带与可见性加权。各分支默认降到 512 维，评价模式拼接并归一化为 3584 维特征。

首次训练从 timm 提供的通用预训练骨干开始微调，分类与条带头由项目初始化命令中的 `--allow-pretrained-download` 明确允许第一次下载；骨干的具体配置由固定版本 timm 决定，可查看：

```bash
.venv/bin/python -c "from timm.models import get_pretrained_cfg; print(get_pretrained_cfg('swin_base_patch4_window7_224'))"
```

预训练权重保存在 timm/相关下载库的缓存中，不是本项目训练成果。训练得到的 `best_model.pth` 和 `latest_checkpoint.pth` 放在 `runs/`。本仓库不提供历史项目权重，读者需要通过训练生成自己的权重。

## 训练

确保图片和划分清单已准备好，`output_dir` 尚不存在，然后执行：

```bash
.venv/bin/python -B training/train_reid.py --config training.local.json --device cpu --allow-pretrained-download
```

训练使用身份采样、SwinReID 网络、组合损失、AdamW 和学习率调度。损失由全局 ArcFace、条带分类、Circle、全局 Triplet 及条带拼接 Triplet 组成，具体权重在 `src/loss.py`。骨干采用较低学习率，按原脚本的计划逐步解冻；默认前十轮学习率预热。

终端会显示 epoch、损失和训练分类准确率。训练分类准确率不是检索 Rank-1。每到 `eval_every`，只在留出的验证身份上使用原图与水平翻转特征进行评价，按验证 Rank-1 保存最佳权重，不执行重排序。

产物位于 `runs/reid-01/`：

| 文件 | 用途 |
| --- | --- |
| `configuration.json` | 实际使用的参数 |
| `metrics.jsonl` | 每轮损失、训练分类准确率与最佳验证指标 |
| `best_model.pth` | 验证 Rank-1 最佳时的模型，用于独立测试和检索 |
| `latest_checkpoint.pth` | 最近完整轮次的模型、损失、优化器、调度器、随机状态等，用于恢复 |

保存的最佳 Rank-1、mAP 和日志中的百分比字段取值范围是 0–100。评价脚本输出则为 0–1，读取时不要混用。

新实验更换 `output_dir`；已有实验不会自动覆盖。若只想先尝试少量轮次，可以在首次启动前降低 `epochs` 并同步令 `eval_every` 不大于它。少量轮次不能代表模型已经充分训练，预热和解冻计划也不会自动按比例缩短。

这条训练命令和下面的恢复命令尚未完成真实数据实测，不能把网络前向检查等同于训练成功。训练是否完成应以无异常结束、检查点生成及后续能严格加载为依据。

## 恢复训练

中断后继续使用相同配置、数据和划分清单：

```bash
.venv/bin/python -B training/train_reid.py --config training.local.json --device cpu --resume runs/reid-01/latest_checkpoint.pth
```

恢复最近保存的完整轮次，未保存轮次会重新计算。恢复不再次下载通用预训练权重。`best_model.pth` 只是最佳模型产物，不能代替完整恢复检查点。

恢复时保持原参数和计划总轮数；已达到总轮数的任务不能直接作为同一次实验继续追加。历史旧检查点缺少当前恢复格式时会报错，不会用 `strict=False` 忽略不匹配。

## 独立测试

训练产生最佳模型后，在项目根目录执行：

```bash
.venv/bin/python -B training/evaluate_reid.py --config training.local.json --split test --weight runs/reid-01/best_model.pth --output outputs/reid-01-test.json --device cpu --tta
```

脚本严格加载权重，使用原始测试 Query/Gallery，输出 JSON，包含分区、样本数、具有正确候选的查询数、Rank-1 和 mAP。输出文件若存在则停止，避免覆盖。需要查看验证集时改用 `--split validation` 并更换输出文件名；验证集成绩不能写成独立测试成绩。

Rank-1 是首位候选身份正确的比例；mAP 衡量正确候选的整体排序。两个指标输出为 0–1。当前规则按原始身份编号相等判断匹配，不执行同摄像头排除或重排序。没有正候选的查询不计入均值；本项目划分检查要求 Query 身份在 Gallery 中存在。

评价要求使用训练时的同一划分清单。比较模型时保持分区、翻转增强和评价规则一致，不用反复查看测试成绩来选择参数。

## 图片检索

先选一个实际存在的 Query 图片，然后用训练得到的最佳权重检索：

```bash
query=$(.venv/bin/python -c "from pathlib import Path; p=next(Path('data/original/BallShow-ReID/query').glob('*.jpg')); print(p)")
.venv/bin/python -B retrieve.py --config training.local.json --weight runs/reid-01/best_model.pth --query "$query" --output outputs/retrieval.json --device cpu
```

若修改了数据根目录，应将第一条命令的 Query 路径同步改为实际位置。`--query` 接受一张球员裁剪图片。

默认计算完整 Gallery。可以追加 `--gallery-limit 12` 仅查看前 12 张图片的流程，但这时不一定存在正确候选，也不能作为完整图库精度。输出 JSON 包含实际图库数量、Top5 文件名和余弦相似度；候选图片在 `data_root/bounding_box_test/` 中。

检索使用单视图 Query，以及原图、翻转、中心裁剪三种 Gallery 特征融合；它与评价脚本的双视图协议不同。相似度不是身份正确的概率，单次 Top5 也不是 Rank-1/mAP 指标。该入口计算真实图像特征，不读取预计算检索结果。

## 常见问题

| 现象 | 排查方式 |
| --- | --- |
| 找不到 src、配置或数据 | 回到项目根目录，核对配置文件相对路径与解压层级 |
| 依赖安装后仍无法导入 | 统一使用 `.venv/bin/python` 安装和执行 |
| 数据划分提示身份或图片内容重叠 | 核对原始数据来源与文件分区，不忽略检查 |
| 划分清单已经存在 | 继续使用该实验的原清单；新实验换清单路径和输出目录 |
| 批量不能包含足够身份 | 保证至少两个身份、每身份至少两张，并满足批量整除规则 |
| 输出目录已经存在 | 恢复时使用 `--resume`，新训练则更换 `output_dir` |
| 预训练下载失败 | 检查上游资源访问与缓存空间；不把历史项目权重冒充通用骨干 |
| 权重结构或类别数不匹配 | 核对网络、训练身份和文件来源，不用非严格加载掩盖错误 |
| 恢复训练报告数据变化 | 使用保存实验时的图片和划分，变更数据应建立新实验 |
| CPU 训练过慢或内存不足 | 降低批量并建立新实验；本教程没有完整训练耗时或硬件保证 |
| 输出 JSON 已存在 | 更换结果文件名，已有结果不会自动覆盖 |

## 测试与使用边界

必要测试入口：

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

测试使用受控样例检查身份编号、采样、指标计算、划分隔离、图片内容变化及随机状态往返。这些规则已有通过记录，打包时只做静态核对；它们不能替代真实数据训练、完整恢复、独立测试或模型精度评价。

本版提供 ReID 训练与检索代码，不包含动作识别、视频追踪、本地页面或服务器服务。模型、数据及运行缓存不随仓库提供。

## 许可证与引用

维护署名：hahawang6。代码采用 [GNU AGPL-3.0](LICENSE)，允许按条款使用、修改和商业使用，适用时须提供相应源码。第三方代码、图片和权重的授权分别适用，不自动继承本项目许可证。引用及数据来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
