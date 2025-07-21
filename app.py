import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator, List, Optional

import pytz
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai

# -----------------------------------------------------------------------
# 0. 配置
# -----------------------------------------------------------------------
shanghai_tz = pytz.timezone("Asia/Shanghai")

credentials = json.load(open("credentials.json"))
API_KEY = credentials["API_KEY"]
BASE_URL = credentials.get("BASE_URL", "")

if API_KEY.startswith("sk-"):
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    USE_GEMINI = False
else:
    import os
    os.environ["GEMINI_API_KEY"] = API_KEY
    gemini_client = genai.Client()
    USE_GEMINI = True

if API_KEY.startswith("sk-REPLACE_ME"):
    raise RuntimeError("请在环境变量里配置 API_KEY")

templates = Jinja2Templates(directory="templates")

# -----------------------------------------------------------------------
# 1. FastAPI 初始化
# -----------------------------------------------------------------------
app = FastAPI(title="AI Animation Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    topic: str
    history: Optional[List[dict]] = None

# -----------------------------------------------------------------------
# 2. 核心：流式生成器 (现在会使用 history)
# -----------------------------------------------------------------------
async def llm_event_stream(
    topic: str,
    history: Optional[List[dict]] = None,
    model: str = "gemini-2.5-pro", # Changed model for better performance if needed
) -> AsyncGenerator[str, None]:
    history = history or []
    
    # The system prompt is now more focused and references a structural example
    fermats_example_structure = """
**请严格参考以下 HTML/CSS/JS 结构来生成动画,但不要模仿内容本身:**

1.  **单一 HTML 文件**: 所有代码（HTML, CSS, JS）都在一个 HTML 文件中。
2.  **CSS 样式**:
    *   在 `<head>` 中使用 `<style>` 标签。
    *   使用 CSS 变量 (e.g., `:root { --bg-color: #...; }`) 来定义颜色和字体方案，方便统一修改。
    *   定义关键帧动画 (`@keyframes`) 用于元素的淡入、淡出等效果。
3.  **HTML 结构**:
    *   主体是一个大的容器 `<div id="animation-container">`，固定分辨率（如 2560x1440），并使用 CSS transform scale 实现响应式布局。
    *   容器内有一个 `<svg id="scene">` 元素，作为所有动画视觉元素的画布。
    *   SVG 内使用 `<g>` 元素来组织不同的场景或对象组。
    *   一个单独的 `<div id="subtitles">` 用于显示中英双语字幕，位于动画容器之外，通过 CSS 定位在底部。
4.  **JavaScript 逻辑**:
    *   所有 JS 代码放在 `<body>` 结尾的 `<script>` 标签内。
    *   **核心是时间线 (Timeline)**: 创建一个名为 `timeline` 的 JavaScript 数组。
    *   `timeline` 数组中的每个元素都是一个对象，代表一个动画步骤，包含两个关键属性: `{ delay: <毫秒>, action: () => { ... } }`。
    *   `delay` 是指与上一个动画步骤结束后的等待时间。
    *   `action` 是一个函数，执行该步骤的具体动画逻辑（如修改SVG元素属性、显示字幕等）。
    *   **辅助函数**: 编写简洁的辅助函数来操作 DOM，例如：
        *   `$` 选择器函数。
        *   `updateSubtitles(cn, en)` 函数来更新字幕内容。
        *   `setOpacity(selector, value)` 等函数来控制动画。
        *   使用 `requestAnimationFrame` 来实现平滑的动态效果（例如，沿着路径移动的动画）。
    *   **主执行函数**: 一个 `startAnimation` 函数，它会:
        *   设置响应式缩放的事件监听器。
        *   遍历 `timeline` 数组，并使用 `setTimeout` 根据累计的 `delay` 来依次执行每个步骤的 `action`。
    *   通过 `window.onload = startAnimation;` 启动整个动画。
"""

    system_prompt = f"""请你根据用户的主题“{topic}”，生成一个精美的、信息丰富的动态动画。
这个动画需要像一个自动播放的短视频，用视觉和旁白清晰地讲解一个知识点。

**输出要求**:
*   **格式**: 生成一个独立的 HTML 文件，包含所有必需的 HTML, CSS, 和 JavaScript。
*   **视觉风格**: 页面设计要极为精美、现代、有设计感。使用和谐的浅色配色方案，丰富的视觉元素，确保知识和图像的准确性。
*   **动画形式**: 动画需要是动态的，能够展示一个过程。
*   **旁白**: 提供中英双语字幕作为旁白。
*   **分辨率**: 所有视觉元素都应在 2K 分辨率 (2560x1440) 的容器内正确定位，避免视觉错误。
*   **无交互**: 动画自动播放，不需要用户交互。

{fermats_example_structure}

现在，请为主题“{topic}”创作动画。"""

    if USE_GEMINI:
        try:
            full_prompt = system_prompt
            if history:
                history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
                full_prompt = history_text + "\n\n" + full_prompt
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: gemini_client.models.generate_content(
                    model="gemini-2.0-flash-exp", 
                    contents=full_prompt
                )
            )
            
            text = response.text
            chunk_size = 50
            
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                payload = json.dumps({"token": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.05)
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": topic},
        ]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.8, 
            )
        except OpenAIError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            if token:
                payload = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.001)

    yield 'data: {"event":"[DONE]"}\n\n'

# -----------------------------------------------------------------------
# 3. 路由 (CHANGED: Now a POST request)
# -----------------------------------------------------------------------
@app.post("/generate")
async def generate(
    chat_request: ChatRequest, # CHANGED: Use the Pydantic model
    request: Request,
):
    """
    Main endpoint: POST /generate
    Accepts a JSON body with "topic" and optional "history".
    Returns an SSE stream.
    """
    accumulated_response = ""  # for caching flow results

    async def event_generator():
        nonlocal accumulated_response
        try:
            async for chunk in llm_event_stream(chat_request.topic, chat_request.history):
                accumulated_response += chunk
                if await request.is_disconnected():
                    break
                yield chunk
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


    async def wrapped_stream():
        async for chunk in event_generator():
            yield chunk

    headers = {
        "Cache-Control": "no-store",
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(wrapped_stream(), headers=headers)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(
        "index.html", {
            "request": request,
            "time": datetime.now(shanghai_tz).strftime("%Y%m%d%H%M%S")})

# -----------------------------------------------------------------------
# 4. 本地启动命令
# -----------------------------------------------------------------------
# uvicorn app:app --reload --host 0.0.0.0 --port 8000


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
