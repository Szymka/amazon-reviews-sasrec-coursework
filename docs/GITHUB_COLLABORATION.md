# GitHub Collaboration

## 仓库权限

建议课程大作业仓库先设为 Private。仓库只保存代码、配置、文档、小样例和轻量结果摘要，不保存完整 raw/processed 数据。

## 分支建议

- `main`：最终稳定版本，只合并经过检查的代码和文档。
- `dev`：日常整合版本。
- `preprocess/<name>`：数据处理相关修改。
- `model/<name>`：模型训练相关修改。
- `eval/<name>`：评估脚本相关修改。
- `report/<name>`：报告、图表和说明文档相关修改。

不要直接向 `main` 提交。每个人在自己的功能分支完成后发 Pull Request。

## 提交前检查

每次提交前确认：

- `.gitignore` 没有被误删或放宽。
- `data/raw/` 和 `data/processed/` 的完整数据没有进入暂存区。
- checkpoint、日志、`wandb/`、`runs/` 没有进入暂存区。
- 文档中的命令仍然使用相对路径或可配置路径。

建议先运行：

```powershell
git status --short
```

如果看到 `data/raw/...` 或 `data/processed/...` 的大文件出现在状态里，应立刻取消暂存，不要提交。

## 提交信息规范

推荐格式：

```text
docs: update data preprocessing guide
feat: add seqrec training script
fix: correct ndcg calculation
chore: update gitignore
```

常用类型：

- `docs`：文档修改
- `feat`：新增功能
- `fix`：修复 bug
- `chore`：工程维护
- `refactor`：重构
- `test`：测试或检查脚本

## Pull Request 流程

1. 从 `dev` 创建自己的功能分支。
2. 完成修改后本地运行必要检查。
3. 发 Pull Request 到 `dev`。
4. 至少一名组员检查代码、路径、数据文件和运行说明。
5. 通过后合并到 `dev`。
6. 课程提交前由组长把稳定版本从 `dev` 合并到 `main`。
