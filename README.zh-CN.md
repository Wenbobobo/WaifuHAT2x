# WaifuHAT2x

面向本地漫画库的 Real-HAT 超分与 JPEG XL 转换流水线。它强调事务安全、可恢复和完整页面
吞吐，而不是只追求单个 GPU kernel 的峰值。English version: [README.md](README.md)。

## 能力与路由

- 只把非 JXL 图片作为超分输入；EXIF 转正后按原图短边路由。
- 短边 `< 1000` 使用 Real-HAT x4 normal；`>= 1000` 使用 sharper。
- 两种路由都先执行原生 x4，再在线性光空间缩到目标短边。
- 默认 `mirror` 模式：源图不变，结果写入独立输出目录。JXL 不会再次进入超分。

| 原图短边 | 模型 |
| --- | --- |
| `< 1000` | Real-HAT-GAN x4 normal |
| `>= 1000` 且需要 SR | Real-HAT-GAN x4 sharper |

999/1000 是有意保留的严格边界。两份 Real-HAT 权重会在启动前一并校验并常驻，避免目录顺序
造成重复加载。

## 安全开始

仓库不包含漫画、权重、凭据或任何本机配置。复制模板后只修改自己的本地文件：

```powershell
Copy-Item config.example.toml config.toml
# 编辑 config.toml，确保 input、output 和 models 指向自己的目录，且 input/output 不重叠。

$env:WAIFUHAT_WSL_DISTRO = "YourDistro" # 可选
.\install.bat
.\inspect_workload.bat
.\run_upscale.bat
```

`inspect_workload.bat` 只读扫描，不加载模型、不启动 GPU、不改写图片。首次必须先用 `mirror`
检查输出和观感。需要多份本地配置时，完整复制模板到 `config.local.toml`，再设置：

```powershell
$env:WAIFUHAT_CONFIG = "config.local.toml"
```

Windows 启动器支持 `WAIFUHAT_WSL_DISTRO` 与 `WAIFUHAT_CONFIG`。若 ROCm 环境必须先加载厂商提供
的 shell 设置，可设置 `WAIFUHAT_ROCM_ENV` 指向该脚本；它不会被仓库记录。

## 权重与替换

权重不进入 Git。安装器根据 [model_sources.toml](model_sources.toml) 下载并校验 SHA-256；只包含
Real-HAT normal/sharper 和 HAT-S x2/x4 的来源信息。请自行阅读上游条款、只从可信来源下载并校验
哈希。PyTorch checkpoint 是反序列化输入，不能用未知文件替换。

`replace` 会在 JXL 解码、尺寸、哈希和最终文件验证全部通过后才删除源图，但它依然具有破坏性。
必须先完成 mirror 验证并保留备份，再在**本地、未提交的**配置中显式打开它。报错后不要手工删除
状态文件、worklist、`.part`、源图或候选 JXL；用相同配置恢复，无法证明安全时系统会保留两份文件。

详细操作见 [运行手册](docs/OPERATIONS.md)。

## 性能与贡献

当前稳定配置是 BF16 eager、tile `[256, 320]`、overlap 16、batch 1 和单 GPU lane。性能变化要先
通过 12 页筛选，再用固定 30 页完整页面墙钟、显存、确定性与盲测 ROI 终审。不要上传样本页、
模型、原始遥测、输出哈希或本机路径。

开发前运行：

```bash
uv lock --check
python -m ruff check src tests scripts
python -m compileall -q src scripts tests
python scripts/check_public_tree.py
python -m pytest -q
```

更多约束见 [CONTRIBUTING.md](CONTRIBUTING.md)、[性能协议](docs/PERFORMANCE.md) 和
[ROCm 运行时说明](docs/ROCM_RUNTIME.md)。项目自身代码采用 [Apache-2.0](LICENSE)，依赖与模型权重
分别遵守其上游许可证和条款。
