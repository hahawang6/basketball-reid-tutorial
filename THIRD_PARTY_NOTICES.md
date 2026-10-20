# 第三方引用

本项目采用 GNU AGPL-3.0，正文见 [LICENSE](LICENSE)。下列依赖通过包管理器安装，不随项目打包；依赖的原始许可保持有效。

| 项目 | 用途 | 上游许可 |
| --- | --- | --- |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) | 张量计算与网络层 | 见上游 LICENSE |
| [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE) | 图像变换与增强 | BSD-3-Clause |
| [timm](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE) | Swin Transformer 骨干实现 | Apache-2.0 |
| [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | 距离矩阵与指标计算 | BSD-3-Clause |
| [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE) | 图像读取与增强 | HPND，具体条款见上游 |

[tqdm](https://github.com/tqdm/tqdm/blob/master/LICENCE) 用于训练进度显示，按其上游许可使用。

BallShow-ReID 的[原始数据入口来源](https://www.fwwb.org.cn/topic/show/a00d22f4-b2c3-4456-828a-b3dc18661b0b)公布了企业网盘。该页面没有明确授予重新打包分发的许可，因此本项目只提供原始获取链接，不附带图片。数据不自动适用本项目代码许可证。

方法参考：

- [Swin Transformer](https://arxiv.org/abs/2103.14030)：层级视觉 Transformer 骨干。
- [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512)：GeM 池化。

维护署名：hahawang6。
