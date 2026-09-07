"""存储层：SQLite 记录联系人/消息/处理周期，并提供仓库与文件存储。

- db            连接与会话/消息/周期基础读写
- contact_control  联系人控制（黑名单 / 移除 / 活动联系人）
- leads_repo    线索仓库（阶段 / 更新时间 / 跟进）
- messages_repo 消息查询仓库
- file_store    导入文档的复制与索引
"""
from __future__ import annotations

from . import db
from . import contact_control
from . import leads_repo
from . import messages_repo
from . import file_store

__all__ = [
    "db",
    "contact_control",
    "leads_repo",
    "messages_repo",
    "file_store",
]