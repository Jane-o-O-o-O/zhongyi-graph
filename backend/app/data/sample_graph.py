from app.models.graph import EvidenceCard, GraphEdge, GraphNode


SAMPLE_NODES = [
    GraphNode(id="symptom:失眠", label="Symptom", name="失眠"),
    GraphNode(id="syndrome:心脾两虚", label="Syndrome", name="心脾两虚"),
    GraphNode(id="treatment:补益心脾", label="Treatment", name="补益心脾"),
    GraphNode(id="formula:归脾汤", label="Formula", name="归脾汤"),
    GraphNode(id="herb:党参", label="Herb", name="党参"),
    GraphNode(id="formula:柴胡桂枝干姜汤", label="Formula", name="柴胡桂枝干姜汤"),
    GraphNode(id="herb:柴胡", label="Herb", name="柴胡"),
    GraphNode(id="herb:桂枝", label="Herb", name="桂枝"),
    GraphNode(id="herb:干姜", label="Herb", name="干姜"),
    GraphNode(id="indication:往来寒热", label="Indication", name="往来寒热"),
]

SAMPLE_EDGES = [
    GraphEdge(
        id="edge:insomnia:syndrome",
        source="symptom:失眠",
        target="syndrome:心脾两虚",
        relation="MANIFESTS_AS",
        display="可辨为",
        evidence_ids=["evidence:insomnia:1"],
    ),
    GraphEdge(
        id="edge:syndrome:treatment",
        source="syndrome:心脾两虚",
        target="treatment:补益心脾",
        relation="RECOMMENDS_TREATMENT",
        display="治法",
        evidence_ids=["evidence:insomnia:1"],
    ),
    GraphEdge(
        id="edge:treatment:formula",
        source="treatment:补益心脾",
        target="formula:归脾汤",
        relation="RECOMMENDS_FORMULA",
        display="推荐方剂",
        evidence_ids=["evidence:insomnia:2"],
    ),
    GraphEdge(
        id="edge:guipi:dangshen",
        source="formula:归脾汤",
        target="herb:党参",
        relation="COMPOSED_OF",
        display="组成",
        evidence_ids=["evidence:insomnia:2"],
    ),
    GraphEdge(
        id="edge:chaihu:indication",
        source="formula:柴胡桂枝干姜汤",
        target="indication:往来寒热",
        relation="TREATS",
        display="主治",
        evidence_ids=["evidence:formula:1"],
    ),
    GraphEdge(
        id="edge:chaihu:herb1",
        source="formula:柴胡桂枝干姜汤",
        target="herb:柴胡",
        relation="COMPOSED_OF",
        display="组成",
        evidence_ids=["evidence:formula:1"],
    ),
    GraphEdge(
        id="edge:chaihu:herb2",
        source="formula:柴胡桂枝干姜汤",
        target="herb:桂枝",
        relation="COMPOSED_OF",
        display="组成",
        evidence_ids=["evidence:formula:1"],
    ),
    GraphEdge(
        id="edge:chaihu:herb3",
        source="formula:柴胡桂枝干姜汤",
        target="herb:干姜",
        relation="COMPOSED_OF",
        display="组成",
        evidence_ids=["evidence:formula:1"],
    ),
]

SAMPLE_EVIDENCE = [
    EvidenceCard(
        id="evidence:insomnia:1",
        title="失眠证候线索",
        source="本地中医知识库",
        snippet="失眠可围绕心脾两虚、肝郁化火、阴虚火旺等证候展开分析。",
        source_type="local",
        location="seed://insomnia",
    ),
    EvidenceCard(
        id="evidence:insomnia:2",
        title="方剂推荐线索",
        source="本地方剂资料",
        snippet="心脾两虚型失眠常围绕补益心脾、养血安神方向组织方药。",
        source_type="local",
        location="seed://formula",
    ),
    EvidenceCard(
        id="evidence:formula:1",
        title="柴胡桂枝干姜汤条目",
        source="中国方剂数据库",
        snippet="柴胡桂枝干姜汤，和解少阳，兼化痰饮，组成含柴胡、桂枝、干姜等。",
        source_type="local",
        location="TCM-DB/zyfj.csv",
    ),
]
