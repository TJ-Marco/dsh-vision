# 贡献指南

感谢你对 dsh-vision 的兴趣！请遵循以下约定，让协作更顺畅。

## 提交信息规范（Conventional Commits）

采用 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/) 格式：

```
<type>(<scope>): <描述>

[正文：说明改动动机与验证方式]
```

常用 `type`：

| type | 含义 |
|---|---|
| `feat` | 新功能 |
| `fix` | 缺陷修复 |
| `refactor` | 重构（行为不变） |
| `test` | 测试相关 |
| `docs` | 文档 |
| `ci` / `build` | CI / 构建配置 |
| `chore` | 杂项（依赖、格式化等） |

`scope` 可省略；如涉及具体模块可写，例如 `feat(video): 支持场景切换抽帧`。

示例：

```
feat(api): 新增 /api/face/verify 端点

- 基于注册表的 embedding 做余弦相似度比对
- 新增 tests/test_face_verify.py（5 个用例）
```

## 版本标签（SemVer）

发布版本使用语义化版本号，格式 `v<major>.<minor>.<patch>`，例如：

```bash
git tag v0.1.0
git push origin v0.1.0
```

- 打 `v*` 标签会触发 GitHub Actions 自动构建并推送 Docker 镜像（ghcr.io）。
- 破坏性 API 变更 → `major`；新增向后兼容功能 → `minor`；缺陷修复 → `patch`。
- 0.x 阶段：`minor` 可视为破坏性变更（0.1 → 0.2 允许不兼容）。

## 发布检查清单

- [ ] `ruff check app scripts tests examples` 通过
- [ ] `ruff format app scripts tests examples --check` 通过
- [ ] `python -m pytest tests` 全绿
- [ ] `scripts/download_models.py` 可下载各模型（至少目标检测）
- [ ] Docker 镜像可构建、`/health` 返回 200
- [ ] README 的「模型许可列表」与 `models.json` 一致
- [ ] 更新版本号（`app/__init__.py` 与标签一致）
- [ ] 打标签：`git tag v0.1.0 && git push origin v0.1.0`

## 仓库简介（GitHub description）

一句话简介（发布时填写到仓库设置）：

> 供 DeepSeek 通过 function calling 调用的开源视觉识别服务：图像与视频的目标检测、描述、场景、OCR 与人脸识别，FastAPI + Python SDK + Docker 一键部署。
