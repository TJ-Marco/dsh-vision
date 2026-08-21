"""时间线聚合：把逐帧识别结果压缩为"事件"。

对每个目标 label 做游程归并：同一 label 连续出现（帧索引相邻）合并为一个
事件 {event, start_sec, end_sec, max_confidence}；帧索引断档即结束上一段。
MVP 只聚合目标检测的 label；OCR/人脸仅逐帧返回（后续可扩展）。
"""

from __future__ import annotations


def build_timeline(frame_results: list[dict], timestamps: list[float]) -> list[dict]:
    """由逐帧结果与时间戳生成事件时间线。

    :param frame_results: pipeline.run 的逐帧结果（含 "objects" 列表）
    :param timestamps: 与帧一一对应的时间戳（秒）
    :return: 按出现顺序排列的事件列表
    """
    active: dict[str, dict] = {}
    completed: list[tuple[str, dict]] = []

    for index, result in enumerate(frame_results):
        # 本帧各 label 的最大置信度
        labels: dict[str, float] = {}
        for obj in result.get("objects", []):
            label = obj["label"]
            labels[label] = max(labels.get(label, 0.0), float(obj["confidence"]))

        for label, confidence in labels.items():
            run = active.get(label)
            if run is not None and run["end"] == index - 1:
                run["end"] = index
                run["max_conf"] = max(run["max_conf"], confidence)
            else:
                active[label] = {"start": index, "end": index, "max_conf": confidence}

        # 本帧未出现且上一段恰好在上一帧结束的 label → 收尾
        for label in [name for name, run in active.items() if run["end"] == index - 1 and name not in labels]:
            completed.append((label, active.pop(label)))

    for label, run in active.items():
        completed.append((label, run))

    return [
        {
            "event": label,
            "start_sec": round(timestamps[run["start"]], 3),
            "end_sec": round(timestamps[run["end"]], 3),
            "max_confidence": round(run["max_conf"], 4),
        }
        for label, run in sorted(completed, key=lambda item: item[1]["start"])
    ]
