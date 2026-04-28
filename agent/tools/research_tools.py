from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote_plus

import httpx

from agent.tools.common import json_ready


async def web_search_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    query = args.get("query", "")
    if not query:
        return json_ready({"error": "query is required"}), False
    url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            payload = (await client.get(url)).json()
        results = []
        for topic in payload.get("RelatedTopics", [])[:8]:
            if "Text" in topic:
                results.append({"title": topic.get("Text"), "url": topic.get("FirstURL")})
        abstract = payload.get("AbstractText")
        return json_ready({"query": query, "abstract": abstract, "results": results}), True
    except Exception as exc:
        return json_ready({"error": str(exc), "query": query}), False


async def paper_search_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    query = args.get("query", "")
    if not query:
        return json_ready({"error": "query is required"}), False
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": int(args.get("limit", 5)),
                    "fields": "title,year,abstract,url,citationCount,authors",
                },
            )
            response.raise_for_status()
            payload = response.json()
        return json_ready({"query": query, "papers": payload.get("data", [])}), True
    except Exception as exc:
        return json_ready({"error": str(exc), "query": query}), False


async def docs_search_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    library = args.get("library", "")
    query = args.get("query", "")
    if not library or not query:
        return json_ready({"error": "library and query are required"}), False
    search_query = f"{library} {query} site:huggingface.co/docs OR site:modal.com/docs OR site:stable-baselines3.readthedocs.io"
    return await web_search_handler({"query": search_query})


async def docs_fetch_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    url = args.get("url", "")
    if not url:
        return json_ready({"error": "url is required"}), False
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            text = (await client.get(url)).text
        return json_ready({"url": url, "content": text[:20_000]}), True
    except Exception as exc:
        return json_ready({"error": str(exc), "url": url}), False


async def github_find_examples_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    query = args.get("query") or " ".join(
        str(part) for part in [args.get("repo"), args.get("keyword")] if part
    )
    if not query:
        return json_ready({"error": "query or repo/keyword is required"}), False
    gh_query = f"{query} language:Python"
    try:
        async with httpx.AsyncClient(timeout=20, headers={"Accept": "application/vnd.github+json"}) as client:
            response = await client.get(
                "https://api.github.com/search/code",
                params={"q": gh_query, "per_page": int(args.get("limit", 5))},
            )
            if response.status_code == 401 or response.status_code == 403:
                return json_ready(
                    {
                        "query": gh_query,
                        "error": "GitHub code search requires authentication or was rate limited.",
                    }
                ), False
            response.raise_for_status()
            payload = response.json()
        items = [
            {
                "name": item.get("name"),
                "path": item.get("path"),
                "repository": item.get("repository", {}).get("full_name"),
                "html_url": item.get("html_url"),
            }
            for item in payload.get("items", [])
        ]
        return json_ready({"query": gh_query, "results": items}), True
    except Exception as exc:
        return json_ready({"error": str(exc), "query": gh_query}), False


async def github_read_file_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    repo = args.get("repo")
    path = args.get("path")
    ref = args.get("ref", "main")
    if not repo or not path:
        return json_ready({"error": "repo and path are required"}), False
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(url)
            response.raise_for_status()
        return json_ready({"repo": repo, "path": path, "ref": ref, "content": response.text[:30_000]}), True
    except Exception as exc:
        return json_ready({"error": str(exc), "repo": repo, "path": path, "ref": ref}), False


async def hf_repo_files_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    repo_id = args.get("repo_id")
    repo_type = args.get("repo_type", "model")
    path = args.get("path")
    if not repo_id:
        return json_ready({"error": "repo_id is required"}), False
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        if path:
            local = hf_hub_download(repo_id=repo_id, filename=path, repo_type=repo_type)
            from pathlib import Path

            return json_ready(
                {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "path": path,
                    "content": Path(local).read_text(encoding="utf-8", errors="replace")[:30_000],
                }
            ), True
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        return json_ready({"repo_id": repo_id, "repo_type": repo_type, "files": files[:500]}), True
    except Exception as exc:
        return json_ready({"error": str(exc), "repo_id": repo_id, "repo_type": repo_type}), False


async def research_handler(args: dict[str, Any], session: Any = None, **_: Any) -> tuple[str, bool]:
    task = args.get("task", "")
    if not task:
        return json_ready({"error": "task is required"}), False
    # Lightweight v1 research aggregator: enough to ground the main agent without
    # introducing a second autonomous loop yet.
    docs = await docs_search_handler({"library": args.get("library", "reinforcement learning"), "query": task})
    web = await web_search_handler({"query": task})
    papers = await paper_search_handler({"query": task, "limit": 5})
    payload = {
        "task": task,
        "docs": json.loads(docs[0]),
        "web": json.loads(web[0]),
        "papers": json.loads(papers[0]),
        "note": "V1 research aggregates docs/web/papers; deeper citation crawling can be added as a future research sub-agent.",
    }
    return json_ready(payload), True
