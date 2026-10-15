"""Generate the two architecture diagrams as editable draw.io files.

Run with the dedicated venv so drawpyo does not leak into the project deps:

    python -m venv .diagram-venv && .diagram-venv/bin/pip install drawpyo
    .diagram-venv/bin/python scripts/generate_diagrams.py

Then rasterize with the draw.io exporter:

    docker run --rm -v "$PWD/docs":/data -w /data rlespinasse/drawio-export \
      -f svg -o . --output-mode relative --remove-page-suffix .
    docker run --rm -v "$PWD/docs":/data -w /data rlespinasse/drawio-export \
      -f png -o . --output-mode relative --remove-page-suffix .

    Do not pass -t/--transparent to the PNG export: it produces a
    transparent background instead of the white one the README expects.

Finally patch a real white background into the exported SVGs (GitHub's HTML
sanitizer strips the `style="background:#fff"` the exporter writes on the
`<svg>` root, but it does not strip an ordinary `<rect>` element), with:

    python scripts/patch_svg_background.py docs/aws-architecture.svg \
        docs/recommendation-architecture.svg
"""

from __future__ import annotations

from pathlib import Path

import drawpyo
from drawpyo.diagram import Edge

DOCS = Path(__file__).resolve().parents[1] / "docs"

AWS_SHAPE = (
    "sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#{fill};"
    "strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;"
    "align=center;html=1;fontSize=11;fontStyle=0;aspect=fixed;"
    "shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.{icon};"
)
BOX = (
    "rounded=1;whiteSpace=wrap;html=1;fillColor=#{fill};strokeColor=#{stroke};"
    "fontSize=11;fontColor=#232F3E;align=center;verticalAlign=middle;"
)

ORANGE, BLUE, GREEN, PURPLE = "D45B07", "4D72F3", "277116", "8C4FFF"


def aws_node(page, title, icon, fill, x, y):
    node = drawpyo.diagram.Object(page=page, value=title)
    node.apply_style_string(AWS_SHAPE.format(fill=fill, icon=icon))
    node.position = (x, y)
    node.width, node.height = 78, 78
    return node


def box(page, title, x, y, width=190, height=58, fill="FFFFFF", stroke="9AA5B1"):
    node = drawpyo.diagram.Object(page=page, value=title)
    node.apply_style_string(BOX.format(fill=fill, stroke=stroke))
    node.position = (x, y)
    node.width, node.height = width, height
    return node


def link(page, source, target, label=""):
    """Connect two nodes.

    drawpyo's Edge has no apply_style_string, so styling is passed through the
    constructor instead.
    """
    return Edge(
        page=page,
        source=source,
        target=target,
        label=label,
        fontSize=10,
        fontColor="#4A5568",
        strokeColor="#4A5568",
        endArrow="blockThin",
        rounded=1,
    )


def build_aws_architecture() -> None:
    """Offline training on the left, online serving on the right."""
    file = drawpyo.File(file_name="aws-architecture.drawio", file_path=str(DOCS))
    page = drawpyo.Page(file=file, name="AWS architecture")

    box(page, "Offline — nightly retraining", 30, 20, 450, 36, fill="FFF4E6", stroke="D45B07")
    box(
        page, "Online — request path (p99 < 50 ms)",
        560, 20, 530, 36, fill="EEF2FF", stroke="4D72F3",
    )

    kaggle = box(page, "Sephora reviews dataset\n(~1M reviews · 8k products)", 30, 100, 200, 58)
    raw = aws_node(page, "S3\nraw", "bucket", GREEN, 300, 90)
    train = aws_node(page, "ECS Fargate\nhybrid WARP training", "machine_learning", ORANGE, 300, 260)
    schedule = aws_node(page, "EventBridge\nnightly 04:00", "event_time_based", PURPLE, 90, 260)
    artifacts = aws_node(page, "S3\nartifacts", "bucket", GREEN, 300, 430)
    topn = aws_node(page, "DynamoDB\ntop-N lists", "dynamodb", BLUE, 600, 430)

    client = aws_node(page, "Client", "client", BLUE, 600, 90)
    api = aws_node(page, "API Gateway", "api_gateway", PURPLE, 820, 90)
    lam = aws_node(page, "Lambda\ninference", "lambda_function", ORANGE, 820, 260)
    logs = aws_node(page, "CloudWatch\nlogs & metrics", "cloudwatch_2", PURPLE, 1010, 430)

    link(page, kaggle, raw, "ingest")
    link(page, schedule, train, "cron")
    link(page, raw, train, "read")
    link(page, train, artifacts, "feature matrices, embeddings")
    link(page, artifacts, topn, "publish top-N")
    link(page, client, api, "GET /recommend")
    link(page, api, lam, "invoke")
    link(page, lam, topn, "GetItem")
    link(page, lam, logs, "emit")

    file.write()
    print(f"wrote {DOCS / 'aws-architecture.drawio'}")


def build_recommendation_architecture() -> None:
    """The modelling pipeline, from raw reviews to a ranked list.

    Laid out as a single vertical spine with side branches so that no edge
    has to cross a node box.
    """
    file = drawpyo.File(file_name="recommendation-architecture.drawio", file_path=str(DOCS))
    page = drawpyo.Page(file=file, name="Recommendation architecture")

    spine_x = 420
    w, gap = 260, 120

    raw = box(page, "Raw reviews\nauthor · product · rating", spine_x, 40, w, 58, fill="F7FAFC")
    implicit = box(
        page,
        "Implicit feedback\nrating ≥ 4 or is_recommended → positive\nlow ratings dropped",
        spine_x, 40 + gap, w, 72, fill="FFF4E6", stroke="D45B07",
    )
    core = box(
        page, "Iterative k-core filter\n≥5 users / ≥5 items",
        spine_x, 40 + 2 * gap, w, 58, fill="FFF4E6", stroke="D45B07",
    )
    split = box(
        page,
        "Cold-start split\nusers/items held out entirely\n+ leave-last-one-out on the rest",
        spine_x, 40 + 3 * gap, w, 72, fill="FFF4E6", stroke="D45B07",
    )
    warp = box(
        page,
        "WARP triplet sampler\nsample j until s(u,j) > s(u,i) − 1\nweight = L(rank(i))",
        spine_x, 40 + 4 * gap, w, 72, fill="E6FFFA", stroke="277116",
    )
    hybrid = box(
        page,
        "Hybrid feature-sum MF\np_u = Σ e_f,  q_i = Σ e_g\ns(u,i) = ⟨p_u, q_i⟩ + b_i",
        spine_x, 55 + 5 * gap, w, 86, fill="EEF2FF", stroke="4D72F3",
    )
    evaluation = box(
        page,
        "Evaluation\nNDCG/Recall/MAP@10\n+ dedicated cold-start slice",
        spine_x, 70 + 6 * gap, w, 72, fill="FFF5F5", stroke="C53030",
    )

    features = box(
        page,
        "Feature store (the representation)\nconf/features.yaml\nskin type/tone, brand, category,\nprice bucket, ingredient tags",
        spine_x - 340, 40 + 4 * gap - 10, 260, 92, fill="E6FFFA", stroke="277116",
    )
    baselines = box(
        page, "Baselines\nPopularity · ID-only MF (BPR) · Random",
        spine_x - 340, 70 + 6 * gap + 10, 260, 58, fill="F7FAFC",
    )
    serving = box(
        page,
        "Top-N publication\nDynamoDB · parity-tested\n+ request-time feature fallback",
        spine_x + 340, 55 + 5 * gap + 14, 260, 72, fill="EEF2FF", stroke="4D72F3",
    )

    link(page, raw, implicit)
    link(page, implicit, core)
    link(page, core, split)
    link(page, split, warp)
    link(page, warp, hybrid, "SGD")
    link(page, hybrid, evaluation)
    link(page, core, features, "train only, point-in-time")
    link(page, features, warp)
    link(page, baselines, evaluation)
    link(page, hybrid, serving, "top-N + fallback")

    file.write()
    print(f"wrote {DOCS / 'recommendation-architecture.drawio'}")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    build_aws_architecture()
    build_recommendation_architecture()
