"""
Logging utilities for PrepMind application.
Tracks response time, similarity scores, and token usage.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List
import os

# Create logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
log_file = os.path.join(LOG_DIR, f"prepmind_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PrepMind")


class RequestLogger:
    """Context manager for logging request metrics."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting: {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        if exc_type is None:
            logger.info(f"Completed: {self.operation_name} in {duration:.2f}s")
        else:
            logger.error(f"Failed: {self.operation_name} after {duration:.2f}s - {exc_val}")
        return False  # Don't suppress exceptions
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def log_query_metrics(
    query: str,
    mode: str,
    subject: str,
    response_time: float,
    similarity_scores: List[float],
    num_tokens: int = 0,
    error: str = None
) -> None:
    """
    Log metrics for a query execution.
    
    Args:
        query: User's question
        mode: Exam or Placement mode
        subject: Selected subject
        response_time: Time taken to generate response (in seconds)
        similarity_scores: List of similarity scores from retrieval
        num_tokens: Estimated number of tokens used
        error: Error message if any
    """
    if error:
        logger.error(
            f"Query Failed | Subject: {subject} | Mode: {mode} | "
            f"Query: {query[:50]}... | Error: {error}"
        )
        return
    
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    max_similarity = max(similarity_scores) if similarity_scores else 0.0
    
    logger.info(
        f"Query Success | Subject: {subject} | Mode: {mode} | "
        f"Response Time: {response_time:.2f}s | "
        f"Avg Similarity: {avg_similarity:.4f} | "
        f"Max Similarity: {max_similarity:.4f} | "
        f"Sources Retrieved: {len(similarity_scores)} | "
        f"Tokens Used: ~{num_tokens}"
    )


def log_ingestion_metrics(
    source: str,
    source_type: str,
    num_chunks: int,
    success: bool,
    error: str = None
) -> None:
    """
    Log metrics for data ingestion.
    
    Args:
        source: URL or filename
        source_type: PDF or URL
        num_chunks: Number of text chunks created
        success: Whether ingestion was successful
        error: Error message if any
    """
    if error:
        logger.error(
            f"Ingestion Failed | Source: {source} | Type: {source_type} | "
            f"Error: {error}"
        )
        return
    
    if success:
        logger.info(
            f"Ingestion Success | Source: {source} | Type: {source_type} | "
            f"Chunks Created: {num_chunks}"
        )
    else:
        logger.warning(
            f"Ingestion Skipped | Source: {source} | Type: {source_type} | "
            f"Reason: No content extracted"
        )


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a given text.
    Uses a simple approximation: ~4 characters per token.
    """
    return len(text) // 4


def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a summary of recent metrics from the log file.
    """
    # This could be extended to parse the log file and return statistics
    return {
        "logging_enabled": True,
        "log_directory": LOG_DIR,
        "log_file": log_file
    }
