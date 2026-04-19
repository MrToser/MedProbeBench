# -*- coding: utf-8 -*-
"""The main entry point of the Deep Research agent example."""
import asyncio
import os
from dotenv import load_dotenv

from agentscope import logger
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.preprocess import (
    parser_args,
)

from eval_utils import eval_guidebench

async def main() -> None:
    
    args = parser_args()
    logger.setLevel("INFO")
    logger.info(f"os.getenv('TAVILY_API_KEY'):{os.getenv('TAVILY_API_KEY')}")
    pred_path = args.data_path
    await eval_guidebench(args,pred_path)
    
    
if __name__ == "__main__":
    load_dotenv(override=True)
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception(e)
