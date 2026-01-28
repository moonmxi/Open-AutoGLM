# Shared prompt rules and constraints to ensure consistency between System Prompt and User Prompt.

# 1. Anchor Rules (锚点规则)
ANCHOR_RULES_CONTENT = (
    "【场景辨析原则】\n"
    "1. **Tab选中状态（死刑判决）**：必须检查底部TabBar哪个图标是**高亮/彩色**的！\n"
    "2. **中间区域特征（关键证据）**：\n"
    "   - 若中间区域特征不符，即使底部Tab长得一样，也判定为**不同场景**。\n"
    "3. **动态内容过滤**：忽略具体商品图、用户名、数字。只看“有没有这个板块”。\n"
    "4. **前后文辅助**：必须结合 before/after 截图的差异来判断动作意图。"
)

# 2. Per-action Protocol (单步执行协议)
ACTION_PROTOCOL_CONTENT = (
    "1) **找茬模式**：在执行动作前，先找出当前屏幕与 before 截图的 **3处不同点**。\n"
    "   - 如果不同点仅是文字/图片内容（如商品名变了），则视为匹配。\n"
    "   - 如果不同点是**功能入口/布局结构/Tab选中项等**（如‘搜索框’变成了‘头像’），则视为**不匹配**。\n"
    "2) **导航补救**：若 Launch 后默认在首页，而 before 截图非首页，**必须先执行导航 ** 过去，严禁直接执行后续动作！\n"
    "3) **执行与核验**：只有场景严格匹配时才执行动作。执行后确认是否引发了预期的画面变化。"
)

# 3. Screenshot Policy (截图策略)
SCREEN_POLICY_CONTENT = (
    "微流程：默认使用 3 张等权帧（必要时补第 4 帧）。"
    "内部索引仅用于排序，不参与重要性判断；所有时间线帧等权使用。"
    "每条动作必须在与该动作对应的截图场景中执行；屏幕不匹配先 Back/Home/Launch/导航回去。"
    "执行后确认是否进入该动作的 after 截图，再继续。"
)

# 4. Timeline Focus (时间线聚焦)
TIMELINE_FOCUS_CONTENT = (
    "同等重视微流程中的每一帧。\n"
    "对于每一个动作，必须使用该步骤对应的 before/after 截图进行对比；\n"
    "特别注意：Launch 动作通常会打开 APP 的默认首页（如 Tab 1）。如果目标动作需要在其他 Tab执行，**必须先导航**，不要假设 Launch 后就直接到了目标页！\n"
    "严格按顺序复现：帧 -> 动作 -> 下一帧。"
)

ACTION_FORMAT_RETRY_CONTENT = (
    "你上一条输出的 <answer> 未包含可执行动作。"
    "请立刻重新输出：<answer> 中只允许包含 1 条动作，且必须是 do(...) 或 finish(...)，"
    "前后不要有任何解释性文字、编号、冒号、Markdown。"
)
