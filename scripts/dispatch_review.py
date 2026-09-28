#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cross-review 路线 A 自动分发器：把评审任务卡发给多个不同家族的大模型，回收完整原文。

用法：
  python dispatch_review.py --task-file 任务卡.md \
      --models "qwen3.5-plus,deepseek-v4.1-flash,kimi-k2.6" \
      --out-dir 回收目录

前提：一个 OpenAI 兼容端点的 API Key，按以下顺序取：
  1) 环境变量（名由 --api-key-env 指定，默认 DASHSCOPE_API_KEY）
  2) ~/.aliyun/config.json 当前 profile 的 dashscope.api_key（阿里云百炼场景，标准 sk- 开头；
     Coding Plan 的 sk-sp- 不支持）
  注意：按量计费，每次分发前须取得用户本轮授权。

可选参数：
  --base-url      API 入口，默认百炼 OpenAI 兼容模式（换别家 OpenAI 兼容服务改这一项）
  --api-key-env   读哪个环境变量取密钥，默认 DASHSCOPE_API_KEY
  --max-tokens    回复上限，默认 8192（评审长文档时可调大）
  --temperature   采样温度，默认 0.3（评审要求可复现，不建议调高）
  --timeout       单模型超时秒数，默认 300
  --list-models   只列出当前密钥可用的模型清单后退出（诊断用，不发生成费用）；
                  某家报 404/400 时先用它核对模型名是否已下线改名

产出：
  {out-dir}/{模型名}.md —— 每家一份完整原文，文件头带分发元数据（模型/时间/token用量），
  可直接作为 A3 对比的输入与 A5 的原始回复存档。

退出码：全部成功 0；任一模型失败 1（已成功的各家文件不受影响）。
"""
import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"


def resolve_api_key(env_name=DEFAULT_API_KEY_ENV):
    """按优先级取密钥：aliyun CLI 配置 -> 环境变量（名由 --api-key-env 指定）。只返回值，不打印。"""
    cfg = Path.home() / ".aliyun" / "config.json"
    if cfg.exists():
        try:
            d = json.loads(cfg.read_text(encoding="utf-8"))
            current = d.get("current", "default")
            for p in d.get("profiles", []):
                if p.get("name") == current:
                    k = (p.get("dashscope") or {}).get("api_key")
                    if k:
                        return k, f"~/.aliyun/config.json (profile: {current})"
        except Exception:
            pass
    k = os.environ.get(env_name)
    if k:
        return k, f"env {env_name}"
    sys.exit(
        f"[dispatch] 找不到 API Key。请配置以下任一项：\n"
        f"  1) 环境变量 {env_name}（不想用这个名字就用 --api-key-env 改）\n"
        "  2) ~/.aliyun/config.json 当前 profile 的 dashscope.api_key（阿里云百炼）\n"
        f"默认端点 {DEFAULT_BASE_URL}；换 OpenAI 兼容的其他厂商用 --base-url。"
    )


def chat(base_url, key, model, messages, max_tokens, temperature, timeout):
    """调用一次对话接口，带一次重试（限流/网络抖动）。失败抛 RuntimeError。"""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    last_err = None
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions",
                data=body,
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "ignore")[:300]
            except Exception:
                pass
            last_err = f"HTTP {e.code}: {detail}"
            if e.code in (429, 500, 502, 503, 504) and attempt == 1:
                time.sleep(5)
                continue
            raise RuntimeError(last_err)
        except Exception as e:
            last_err = str(e)
            if attempt == 1:
                time.sleep(3)
                continue
            raise RuntimeError(last_err)
    raise RuntimeError(last_err or "unknown error")


def list_models(base_url, key, timeout):
    """列出当前密钥可用的模型，按家族归组打印。失败抛 RuntimeError。"""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + key},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    ids = sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))
    fams = {}
    for i in ids:
        fams.setdefault(i.split("-")[0], []).append(i)
    print(f"[dispatch] 可用模型共 {len(ids)} 个，{len(fams)} 个家族：")
    for f, lst in sorted(fams.items()):
        print(f"  [{f}] {len(lst)} 个: {', '.join(lst)}")


def main():
    ap = argparse.ArgumentParser(description="cross-review 多模型评审分发器")
    ap.add_argument("--task-file", help="A1 任务卡路径（全文原样发给每一家；--list-models 模式下不需要）")
    ap.add_argument("--models", help="逗号分隔的模型名，须跨家族（如 qwen3.5-plus,deepseek-v4.1-flash,kimi-k2.6）")
    ap.add_argument("--out-dir", help="回收目录（每家一个 .md，已存在则覆盖同名）")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API 入口，默认 {DEFAULT_BASE_URL}")
    ap.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV,
                    help=f"读哪个环境变量取密钥，默认 {DEFAULT_API_KEY_ENV}（同时兼容 ~/.aliyun/config.json）")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--list-models", action="store_true",
                    help="只列出当前密钥可用的模型后退出（诊断用，不发生成费用）")
    args = ap.parse_args()

    key, key_source = resolve_api_key(args.api_key_env)
    print(f"[dispatch] 密钥来源: {key_source}")

    if args.list_models:
        try:
            list_models(args.base_url, key, min(args.timeout, 60))
        except Exception as e:
            sys.exit(f"[dispatch] 模型清单获取失败: {e}")
        return

    if not (args.task_file and args.models and args.out_dir):
        sys.exit("[dispatch] 缺少必填参数：正常分发需要 --task-file / --models / --out-dir 三者齐备"
                 "（或改用 --list-models 查看可用模型）")

    task_path = Path(args.task_file)
    if not task_path.exists():
        sys.exit(f"[dispatch] 任务卡不存在: {task_path}")
    task_text = task_path.read_text(encoding="utf-8")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) < 2:
        print("[dispatch] 警告：只发了 1 家，交叉验证需要 >=2 家不同家族", file=sys.stderr)

    results = []
    for m in models:
        print(f"[dispatch] -> {m} 发送中（任务卡 {len(task_text)} 字符）...")
        t0 = time.time()
        try:
            resp = chat(args.base_url, key, m,
                        [{"role": "user", "content": task_text}],
                        args.max_tokens, args.temperature, args.timeout)
        except Exception as e:
            print(f"[dispatch] <- {m} 失败: {e}")
            results.append((m, False, str(e)))
            continue
        elapsed = time.time() - t0
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        usage = resp.get("usage", {})
        if not content and not reasoning:
            print(f"[dispatch] <- {m} 返回为空（usage={usage}），按失败处理")
            results.append((m, False, "empty response"))
            continue

        fname = m.replace("/", "__") + ".md"
        meta = (
            f"<!-- dispatch metadata\n"
            f"model: {m}\n"
            f"base_url: {args.base_url}\n"
            f"task_file: {task_path}\n"
            f"dispatched_at: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"elapsed_s: {elapsed:.1f}\n"
            f"usage: {json.dumps(usage, ensure_ascii=False)}\n"
            f"finish_reason: {choice.get('finish_reason')}\n"
            f"-->\n\n"
        )
        parts = [meta]
        if reasoning:
            parts.append("> 以下为该模型的思考过程（reasoning_content），评审正文在其后。\n\n"
                         + reasoning + "\n\n---\n\n")
        parts.append(content if content else "(模型仅返回思考过程，无正文)")
        (out_dir / fname).write_text("".join(parts), encoding="utf-8")
        print(f"[dispatch] <- {m} 成功，{len(content)} 字符正文，"
              f"in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')}，"
              f"耗时 {elapsed:.1f}s -> {out_dir / fname}")
        if choice.get("finish_reason") == "length":
            print(f"[dispatch] ⚠️ {m} 回复被 --max-tokens={args.max_tokens} 截断（finish_reason=length），"
                  f"评审结论可能不完整——调大 --max-tokens 重发这一家，或人工检查原文末尾是否戛然而止",
                  file=sys.stderr)
        results.append((m, True, ""))

    print("\n[dispatch] ===== 汇总 =====")
    ok = 0
    for m, succ, err in results:
        if succ:
            ok += 1
            print(f"  OK    {m}")
        else:
            print(f"  FAIL  {m}  ({err[:120]})")
    print(f"[dispatch] 完成：{ok}/{len(models)} 家成功，原文在 {out_dir}")
    sys.exit(0 if ok == len(models) else 1)


if __name__ == "__main__":
    main()
