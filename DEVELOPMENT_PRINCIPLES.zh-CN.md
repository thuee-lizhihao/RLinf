# RLinf 项目代码改动与 Git 版本管理基本原则

本文整理自 `CONTRIBUTING.md` 与 `AGENTS.md`，作为本项目后续代码改动、测试验证、文档更新、提交与 PR 管理的默认原则。

## 最高原则

所有面向用户的变更都必须配套测试和文档，并保证他人能够复现、审查和验证。

代码改动应尽量小而清晰：优先遵循现有目录结构、命名、配置方式和局部代码风格；除非确有必要，不引入额外抽象、不做无关重构、不混入格式化噪声。

## 代码改动原则

- 遵循 Google Python Style Guide，并保持与相邻代码一致。
- 公共类、公共方法和公共函数需要清晰的 Google 风格 docstring。
- 函数和方法参数应添加类型标注；返回类型无法被静态工具可靠推断时，也应显式标注。
- 注释应解释必要的设计意图、复杂逻辑或非显然约束，避免重复代码本身。
- 断言和异常必须带有明确、有帮助的错误信息；尽量在执行除法、索引、远程调用、资源分配等操作前提前检查非法输入和非法状态。
- 使用日志而不是 `print`。`Worker` 内使用 `self.log_info`、`self.log_warning`、`self.log_error`；其他位置使用 `rlinf.utils.logging.get_logger()`。
- 新行为、新接口、新模型、新环境、新算法或修复用户可见 bug 时，应补充单元测试或 e2e 测试。若测试依赖 GPU、真实硬件、大模型或大数据，应在 CI 中合理跳过并在文档或 PR 中说明。
- 性能、训练稳定性、reward 曲线、分布式调度、checkpoint、resume、placement 等敏感路径的改动，需要保留测试结果、运行配置和可复现说明。

## 配置与 YAML 原则

- 新配置优先复制并调整现有配置模板，不从零随意创建。
- YAML 中只写静态值，不写计算字段或动态表达式。
- 用户可配置字段在代码中视为只读，不应被静默覆盖。
- 尽量避免 YAML 字段之间互相引用；确需推导的值应在代码中集中计算和校验。
- 新模型、新环境、新算法、新 runner 或新依赖需要同步更新默认配置、校验逻辑、安装说明、示例和必要测试。

## 验证原则

提交前优先运行与改动范围相匹配的检查：

```bash
pre-commit run --all-files
```

按需运行更小范围的单元测试或 e2e 测试，例如：

```bash
pytest tests/unit_tests
```

如果因为环境、硬件、依赖或耗时限制无法运行完整验证，需要在提交说明或 PR 的测试部分明确写出：已运行的命令、未运行的检查、原因和剩余风险。

## Git 分支原则

- 从最新 `main` 创建功能分支：

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

- 分支名应表达变更目的，推荐使用 `feature/`、`fix/`、`docs/`、`test/`、`refactor/`、`chore/` 等前缀。
- 一个分支聚焦一个主题，避免把无关修复、实验代码、格式化和功能开发混在一起。
- 推送到个人 fork 或远端分支后，通过 PR 合入 `main`，由 CI 和 reviewer 验证后再合并。

## Commit 原则

所有 commit 必须带 `Signed-off-by:`，优先使用：

```bash
git commit -s
```

commit message 遵循 Conventional Commits：

```text
<type>(<scope>): <description>
```

常用类型：

- `feat`: 面向用户的新功能
- `fix`: 面向用户的 bug 修复
- `docs`: 文档变更
- `style`: 纯格式变更，无代码行为变化
- `refactor`: 生产代码重构
- `test`: 新增或重构测试
- `chore`: 构建、依赖、工具或维护性变更

提交标题应使用祈使语气，简洁明确，建议控制在约 72 个字符以内。每个 commit 应保持可审查、可回滚、主题单一。

## PR 原则

- PR 标题使用与 commit 相同的 Conventional Commits 格式。
- PR 描述至少填写 `Description` 和 `Checklist`。
- 若关联 issue，应在 `Motivation and Context` 中链接。
- 若影响训练性能、稳定性、调度策略或 reward 曲线，应在 `How has this been tested?` 中提供测试结果、关键配置和观测结论。
- reviewer 提出问题后，应逐条回应；若建议不清楚或存在分歧，优先讨论清楚再改。

## 代理和协作原则

- 修改前先阅读相关代码和现有文档，理解已有模式后再动手。
- 不回滚他人已有改动；若工作区存在无关改动，应避开它们。
- 对不确定但必须继续推进的地方，添加 `TODO(agent)` 并在说明中记录限制。
- 变更完成后，报告改动范围、验证结果和未验证风险。
