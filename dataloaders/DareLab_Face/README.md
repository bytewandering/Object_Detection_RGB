# DareLab Face DataLoader（RGB）

直接读取 `/home/hzhao/ZHN/datasets/DareLab_Face` 中的原始 PNG，作为
Object_Detection_DCT（DCT 系数输入）的自然图像对照。数据划分、标注和检测器结构与 DCT 版
一致，只有输入表示不同。

## 数据划分

划分写在 COCO 标注文件里，Dataset 使用 `ann_path` 中的全部图片。未指定 `ann_path` 时，
train 读取 `img_path/annotations/train.json`，val 和 test 都读取 `img_path/annotations/val.json`。
这两个文件与 `DareLab_Face_DCT/annotations/` 中的完全相同（两个数据集的 `label.json` 一致）。

| 集合 | 内容 | 帧数 | 有人脸的帧 |
|---|---|---|---|
| train | 每段录像按帧顺序的前 60% | 1441 | 806 |
| val / test | 每段录像的后 40% | 964 | 555 |

| 录像 | 帧数 | train（前 60%） | val/test（后 40%） |
|---|---|---|---|
| rec_20260819_092243（无人脸） | 401 | 240 帧 | 161 帧 |
| rec_20260819_092256（无人脸） | 401 | 240 帧 | 161 帧 |
| rec_20260914_145600 | 801 | 480 帧（人脸 472） | 321 帧（人脸 321） |
| rec_20260914_150101 | 802 | 481 帧（人脸 334） | 321 帧（人脸 234） |

切分点为每段录像的 `int(帧数 × 0.6)`，约第 6 秒（无人脸录像）或第 12 秒处。四段录像都同时
出现在训练和验证中，验证衡量的是同一场景中后续时间段的表现；切分点两侧的相邻帧几乎相同，
但只占很小一部分。

修改划分时需同时修改两个数据集下的 JSON，保持 RGB 与 DCT 实验可比。

## 输出

| 字段 | 说明 |
|---|---|
| img | `[3, H, W]` RGB，`(像素 - 128) / 128`，约在 [-1, 1)；原生 320×240，不缩放、不补边、不做增强 |
| reconstruction_target | 与 img 相同的副本 |
| valid_mask | `[1, H, W]` 全 1 |
| gt_bboxes / gt_bboxes_ignore | 原图像素坐标 xyxy |
| warp_matrix | 单位矩阵 |

归一化与 DCT 版的"量化系数 / 128"对应：像素 128 映射到 0，并正好落在对称 INT8 量化网格上。
检测器在右侧和底部补零到 backbone 总步长（32）的倍数，即 240×320 → 256×320，
三个检测层为 32×40、16×20、8×10，与 DCT 版完全相同。

## 配置示例

```yaml
dataloader:
  train:
    name: DareLabFaceDataLoader
    img_path: /home/hzhao/ZHN/datasets/DareLab_Face
    ann_path: /home/hzhao/ZHN/datasets/DareLab_Face/annotations/train.json
    batch_size: 16
    num_workers: 4
  val:
    name: DareLabFaceDataLoader
    img_path: /home/hzhao/ZHN/datasets/DareLab_Face
    ann_path: /home/hzhao/ZHN/datasets/DareLab_Face/annotations/val.json
    batch_size: 16
    num_workers: 4
```
