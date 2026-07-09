# Structural GraphRAG Chunking Design

## Goal

把现有机械 900 字切片改成结构化 GraphRAG 上游：父级抽取单元用于 LLM 实体/关系抽取，子级检索片段用于 Qdrant 向量召回和证据展示。

## Real Text Findings

本地中医古籍主要是 GB18030/GBK 文本，解析前必须稳健解码。抽样检测到的稳定结构包括：

- `<篇名>`：书名或章节/条目标题。
- `<目录>`：卷、门、类目路径，例如 `卷一\方脉总论`。
- `属性：` 和 `内容：`：正文起点，分别出现在方书、医案、本草类文本中。

代表性样本：

- `074-普济方.txt`：约 551 万字，1907 个 `<篇名>`，1906 个 `<目录>`，正文以 `属性：` 起。
- `013-本草纲目.txt`：约 198 万字，1922 个 `<篇名>`，1921 个 `<目录>`，正文以 `内容：` 起。
- `378-王氏医案绎注.txt`：约 15 万字，章节少但单章很长，需要在句界二次切分。

## Data Model

新增 `extraction_units` 表：

- `unit_id`：父单元主键。
- `source_id`、`page_id`、`unit_index`：来源和顺序。
- `title`、`section_path`、`unit_type`：结构上下文。
- `content`、`char_start`、`char_end`、`metadata`：抽取正文和定位信息。

保留 `document_chunks` 表作为检索子片段，并新增：

- `parent_unit_id`：指向 `extraction_units.unit_id`。
- `unit_index`：父单元序号，方便调试和排序。
- `metadata.section_path`、`metadata.parent_unit_title`：给前端和证据链使用。

## Parser Behavior

1. 解码优先尝试 UTF-8，再尝试 GB18030。
2. 对 `<目录>`、`<篇名>`、`属性：`、`内容：` 做结构扫描。
3. 每个真实章节/条目形成一个父级 `ExtractionUnit`。
4. 长父单元只在句号、分号、换行等自然边界拆分，目标 1500-3000 字，最大约 4000 字。
5. 每个父单元再生成 200-400 字左右的子 `DocumentChunk`，同样优先按句界拆分。
6. LLM 抽取只读取父单元；Qdrant 检索和证据展示读取子片段。

## Compatibility

API 返回保持兼容：上传、任务运行、发布仍使用原有入口。内部保存时同时写入 pages、extraction_units、document_chunks。旧 `document_chunks` 查询接口继续可用。

## Verification

新增测试覆盖：

- 真实 `<目录>/<篇名>/属性/内容` 标记生成父单元。
- 长医案章节按句界拆分，不再硬切固定字数。
- 子 chunk 带 `parent_unit_id` 和完整结构 metadata。
- ingestion service 调用抽取器时传入父单元，不再传子 chunk。
