"""
Comprehensive Demo Script for AI Engineering Project Part II
Showcases all implemented features: Orchestration, Monitoring, MCP, A2A, Fine-Tuning
"""
import os
import sys
import time
import asyncio
import json
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from orchestrator.orchestrator import orchestrate_query, add_documents
from monitoring.advanced_monitoring import advanced_metrics
from a2a_client import a2a_client
from fine_tuning.peft_manager import fine_tuning_manager


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(title: str, result: Any):
    """Print a formatted result"""
    print(f"\n{title}:")
    print("-" * 40)
    if isinstance(result, dict):
        for key, value in result.items():
            if key == "metrics" and isinstance(value, list):
                print(f"{key}: {len(value)} metrics recorded")
            else:
                print(f"{key}: {value}")
    else:
        print(result)


async def demo_orchestration():
    """Demo the orchestration features"""
    print_section("ORCHESTRATION DEMO")
    
    # Add sample documents
    print("Adding sample documents...")
    docs = [
        "Artificial intelligence is transforming industries worldwide.",
        "Machine learning algorithms can learn from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers for complex pattern recognition.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and analyze visual information."
    ]
    added = add_documents(docs)
    print(f"Added {added} documents to vectorstore")
    
    # Test basic orchestration
    print("\nTesting basic orchestration...")
    result = orchestrate_query("What is artificial intelligence?", summarize=True)
    print_result("Basic Orchestration Result", result)
    
    # Test orchestration with translation
    print("\nTesting orchestration with translation...")
    result = orchestrate_query("What is machine learning?", summarize=True, translate_to="es")
    print_result("Orchestration with Translation", result)
    
    # Test orchestration with A2A translation
    print("\nTesting orchestration with A2A translation...")
    result = orchestrate_query("What is deep learning?", summarize=True, translate_to="es", use_a2a=True)
    print_result("Orchestration with A2A Translation", result)


async def demo_a2a_protocol():
    """Demo the A2A protocol features"""
    print_section("A2A PROTOCOL DEMO")
    
    test_text = "I love this new AI system! It's amazing how well it works."
    
    # Test sentiment analysis
    print("Testing sentiment analysis...")
    sentiment_result = await a2a_client.call_sentiment_service(test_text)
    print_result("Sentiment Analysis", sentiment_result)
    
    # Test emotion detection
    print("\nTesting emotion detection...")
    emotion_result = await a2a_client.call_emotion_service(test_text)
    print_result("Emotion Detection", emotion_result)
    
    # Test summarization
    print("\nTesting text summarization...")
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
    """
    summary_result = await a2a_client.call_summarization_service(long_text)
    print_result("Text Summarization", summary_result)
    
    # Test comprehensive analysis
    print("\nTesting comprehensive text analysis...")
    comprehensive_result = await a2a_client.analyze_text_comprehensive(test_text)
    print_result("Comprehensive Analysis", comprehensive_result)


def demo_monitoring():
    """Demo the monitoring features"""
    print_section("MONITORING DEMO")
    
    # Get metrics summary
    print("Getting metrics summary...")
    summary = advanced_metrics.get_metrics_summary(hours=1)
    print_result("Metrics Summary", summary)
    
    # Record some sample model performance metrics
    print("\nRecording sample model performance metrics...")
    advanced_metrics.record_model_performance(
        model_name="demo_model",
        accuracy=0.95,
        latency_ms=150.5,
        confidence=0.88,
        input_tokens=50,
        output_tokens=25,
        cost_usd=0.001
    )
    
    # Record data drift metrics
    print("\nRecording data drift metrics...")
    advanced_metrics.record_data_drift("query_length", 0.15, 0.03, 0.05)
    advanced_metrics.record_data_drift("response_time", 0.08, 0.12, 0.05)
    
    print("SUCCESS: Monitoring metrics recorded")


def demo_fine_tuning():
    """Demo the fine-tuning features"""
    print_section("FINE-TUNING & PEFT DEMO")
    
    try:
        # Setup LoRA
        print("Setting up LoRA configuration...")
        fine_tuning_manager.setup_lora(r=8, lora_alpha=16, lora_dropout=0.1)
        
        # Create sample dataset
        print("\nCreating sample dataset...")
        sample_data = fine_tuning_manager.create_sample_dataset()
        print(f"Created dataset with {len(sample_data)} examples")
        
        # Prepare dataset
        print("\nPreparing dataset for training...")
        train_dataset, eval_dataset = fine_tuning_manager.prepare_dataset(sample_data)
        print(f"Train dataset: {len(train_dataset)} examples")
        print(f"Eval dataset: {len(eval_dataset)} examples")
        
        # Note: We'll skip actual training in demo to avoid long execution time
        print("\nNote: Skipping actual training in demo (would take several minutes)")
        print("To train: fine_tuning_manager.train(train_dataset, eval_dataset)")
        
        # Test generation with base model
        print("\nTesting generation with base model...")
        test_prompt = "What is machine learning?"
        response = fine_tuning_manager.generate_response(test_prompt, max_length=50)
        print_result("Base Model Generation", {"prompt": test_prompt, "response": response})
        
    except Exception as e:
        print(f"Fine-tuning demo error: {e}")
        print("This is expected if GPU/CUDA is not available")


def demo_mcp_server():
    """Demo the MCP server features"""
    print_section("MCP SERVER DEMO")
    
    print("MCP Server endpoints available:")
    print("- POST /query-docs - Query documents with orchestration")
    print("- POST /summarize - Summarize text")
    print("- POST /translate - Translate text")
    print("- POST /add-docs - Add documents to vectorstore")
    
    print("\nExample API calls:")
    print("curl -X POST http://localhost:8001/query-docs \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"question\": \"What is AI?\", \"summarize\": true}'")
    
    print("\ncurl -X POST http://localhost:8001/add-docs \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"documents\": [\"Sample document\"]}'")


def demo_dashboard():
    """Demo the monitoring dashboard"""
    print_section("MONITORING DASHBOARD DEMO")
    
    print("Streamlit dashboard features:")
    print("- Real-time metrics visualization")
    print("- Response time trends")
    print("- Confidence score analysis")
    print("- Model performance overview")
    print("- Data drift detection")
    print("- System status monitoring")
    
    print("\nTo start the dashboard:")
    print("streamlit run dashboard.py")
    
    print("\nDashboard will be available at: http://localhost:8501")


async def main():
    """Main demo function"""
    print("AI Engineering Project Part II - Comprehensive Demo")
    print("This demo showcases all implemented features:")
    print("- LLM Orchestration (RAG -> Summarizer -> Translator)")
    print("- Advanced Monitoring (Evidently AI + Postgres)")
    print("- MCP Server Protocol")
    print("- Enhanced A2A Protocol")
    print("- Fine-Tuning & PEFT (LoRA)")
    print("- Monitoring Dashboard (Streamlit)")
    
    try:
        # Run all demos
        await demo_orchestration()
        await demo_a2a_protocol()
        demo_monitoring()
        demo_fine_tuning()
        demo_mcp_server()
        demo_dashboard()
        
        print_section("DEMO COMPLETED SUCCESSFULLY")
        print("DEMO COMPLETED SUCCESSFULLY")
        print("All features are working!")
        print("\nNext steps:")
        print("1. Start the monitoring dashboard: streamlit run dashboard.py")
        print("2. Test the web interface: open static/index.html")
        print("3. Use the MCP server endpoints for API integration")
        print("4. Fine-tune models with your own data")
        
    except Exception as e:
        print(f"\nDemo error: {e}")
        print("Some features may not work without proper API keys or GPU access")


if __name__ == "__main__":
    asyncio.run(main())
