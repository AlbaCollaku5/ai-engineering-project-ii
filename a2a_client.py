"""Enhanced API-to-API client with multiple AI service integrations."""
import httpx
import asyncio
import time
from typing import Dict, Any, List, Optional
from monitoring.advanced_monitoring import advanced_metrics


class A2AClient:
    """Enhanced API-to-API client for multiple AI services"""
    
    def __init__(self):
        self.services = {
            "sentiment": "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest",
            "emotion": "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base",
            "summarization": "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
            "translation": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-es"
        }
        self.timeout = 30.0
    
    async def call_sentiment_service(self, text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Call sentiment analysis service"""
        start_time = time.time()
        
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.services["sentiment"],
                    json={"inputs": text},
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()
                
                # Process result
                if isinstance(result, list) and len(result) > 0:
                    sentiment_data = result[0]
                    processed_result = {
                        "sentiment": sentiment_data.get("label", "unknown"),
                        "confidence": sentiment_data.get("score", 0.0),
                        "text": text,
                        "service": "huggingface_sentiment"
                    }
                else:
                    processed_result = {"error": "Invalid response format", "raw": result}
                
                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                advanced_metrics.record_metric("a2a_sentiment_latency_ms", latency_ms)
                advanced_metrics.record_model_performance(
                    model_name="huggingface_sentiment",
                    accuracy=processed_result.get("confidence", 0.0),
                    latency_ms=latency_ms,
                    confidence=processed_result.get("confidence", 0.0)
                )
                
                return processed_result
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            advanced_metrics.record_metric("a2a_sentiment_error", 1.0)
            return {"error": str(e), "latency_ms": latency_ms}
    
    async def call_emotion_service(self, text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Call emotion detection service"""
        start_time = time.time()
        
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.services["emotion"],
                    json={"inputs": text},
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()
                
                # Process result
                if isinstance(result, list) and len(result) > 0:
                    emotion_data = result[0]
                    processed_result = {
                        "emotion": emotion_data.get("label", "unknown"),
                        "confidence": emotion_data.get("score", 0.0),
                        "text": text,
                        "service": "huggingface_emotion"
                    }
                else:
                    processed_result = {"error": "Invalid response format", "raw": result}
                
                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                advanced_metrics.record_metric("a2a_emotion_latency_ms", latency_ms)
                
                return processed_result
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            advanced_metrics.record_metric("a2a_emotion_error", 1.0)
            return {"error": str(e), "latency_ms": latency_ms}
    
    async def call_summarization_service(self, text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Call text summarization service"""
        start_time = time.time()
        
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.services["summarization"],
                    json={"inputs": text},
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()
                
                # Process result
                if isinstance(result, list) and len(result) > 0:
                    summary_data = result[0]
                    processed_result = {
                        "summary": summary_data.get("summary_text", ""),
                        "original_length": len(text),
                        "summary_length": len(summary_data.get("summary_text", "")),
                        "compression_ratio": len(summary_data.get("summary_text", "")) / len(text) if text else 0,
                        "service": "huggingface_summarization"
                    }
                else:
                    processed_result = {"error": "Invalid response format", "raw": result}
                
                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                advanced_metrics.record_metric("a2a_summarization_latency_ms", latency_ms)
                
                return processed_result
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            advanced_metrics.record_metric("a2a_summarization_error", 1.0)
            return {"error": str(e), "latency_ms": latency_ms}
    
    async def call_translation_service(self, text: str, target_lang: str = "es", api_key: Optional[str] = None) -> Dict[str, Any]:
        """Call translation service"""
        start_time = time.time()
        
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Update service URL based on target language
            service_url = self.services["translation"]
            if target_lang != "es":
                # For other languages, use a different model
                service_url = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-mul"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    service_url,
                    json={"inputs": text},
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()
                
                # Process result
                if isinstance(result, list) and len(result) > 0:
                    translation_data = result[0]
                    processed_result = {
                        "translation": translation_data.get("translation_text", ""),
                        "original_text": text,
                        "target_language": target_lang,
                        "service": "huggingface_translation"
                    }
                else:
                    processed_result = {"error": "Invalid response format", "raw": result}
                
                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                advanced_metrics.record_metric("a2a_translation_latency_ms", latency_ms)
                
                return processed_result
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            advanced_metrics.record_metric("a2a_translation_error", 1.0)
            return {"error": str(e), "latency_ms": latency_ms}
    
    async def analyze_text_comprehensive(self, text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive text analysis using multiple services"""
        start_time = time.time()
        
        try:
            # Run all analyses in parallel
            tasks = [
                self.call_sentiment_service(text, api_key),
                self.call_emotion_service(text, api_key),
                self.call_summarization_service(text, api_key)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            comprehensive_result = {
                "text": text,
                "analysis": {
                    "sentiment": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
                    "emotion": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
                    "summary": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])}
                },
                "total_latency_ms": (time.time() - start_time) * 1000,
                "services_used": len([r for r in results if not isinstance(r, Exception)])
            }
            
            # Record comprehensive metrics
            advanced_metrics.record_metric("a2a_comprehensive_latency_ms", comprehensive_result["total_latency_ms"])
            advanced_metrics.record_metric("a2a_services_success_rate", comprehensive_result["services_used"] / len(tasks))
            
            return comprehensive_result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            advanced_metrics.record_metric("a2a_comprehensive_error", 1.0)
            return {"error": str(e), "latency_ms": latency_ms}


# Global instance
a2a_client = A2AClient()


# Legacy function for backward compatibility
def call_sentiment_service(text: str, endpoint: str = "https://sentim.example/api/analyze") -> Dict[str, Any]:
    """Legacy function for backward compatibility"""
    try:
        resp = httpx.post(endpoint, json={"text": text}, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}
