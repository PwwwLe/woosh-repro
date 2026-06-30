"""调用本地 Woosh API 生成一段 FLAC 音频的手动测试脚本。

脚本假设 API server 已在 ``http://0.0.0.0:8000`` 启动，并把返回内容写入
``outputs/Woosh-DFlow_API_0.flac``。它是端到端 HTTP smoke test，不包含
pytest 断言。
"""

import httpx
import os
import uuid
import logging
import subprocess
import re
import random
import click
from reapy import reascript_api as RPR

logger = logging.getLogger("sfxfm")
logging.basicConfig(level=logging.INFO)

PORT = 8000
API_URL = f"http://0.0.0.0:{PORT}"
api_url = f"{API_URL}/generate"

def generate(prompt: str, filepath: str) -> str:
    """调用 API，根据文本 prompt 生成音频文件并返回保存路径。"""
    headers = {"Accept": "application/json"}

    data = {
        "version": "0.1",
        "token": "string",
        "args": {
            "model": "Woosh-DFlow",
            "prompt": prompt,
            "cfg": 3.0,
            "sampler": "heun",
            "num_steps": 5,
            "sigma_min": 0.0001,
            "sigma_max": 80,
            "rho": 7,
            "S_churn": 1,
            "S_min": 0,
            "S_noise": 1,
            "guidance_scale": 7.5,
            "noise_scheduler": "karras",
            "seed": random.randint(0, 2**32 - 1),
        },
    }

    # returns FLAC compressed audio
    response = httpx.post(api_url, json=data, headers=headers, timeout=45.0)
   
    # Generate a random filename
    save_dir = os.path.dirname(filepath)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(
        f"Received response from SFXFM API: {response.status_code}, {len(response.content)} bytes"
    )
    # Save the FLAC file
    with open(filepath, "wb") as f:
        f.write(response.content)
        logger.info(f"Saved FLAC file to {filepath}")

    return filepath

generate("car revving", "outputs/Woosh-DFlow_API_0.flac")
