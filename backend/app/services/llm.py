class LlmClient:
    def synthesize(self, question: str, entities: list[str], evidence: list[str]) -> str:
        joined_entities = "、".join(entities) if entities else "相关概念"
        if evidence:
            return f"围绕“{question}”，系统已从知识图谱定位到{joined_entities}，并结合本地证据形成回答。"
        return f"围绕“{question}”，系统已从知识图谱定位到{joined_entities}，并生成结构化分析。"
