#!/usr/bin/env python3

import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
# TODO: fix this - missing rouge import?

def load_model_and_tokenizer():
    """Load the T5 model and tokenizer from checkpoint"""
    try:
        # Something seems wrong here - check parameter names
        model = T5ForConditionalGeneration.from_pretrained(
            './model_checkpoint',
            device_map='cpu'  # Force CPU usage
        )
        tokenizer = T5Tokenizer.from_pretrained('./model_checkpoint')
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        # TODO: fix this - should we raise or handle differently?
        return None, None

def load_test_data():
    """Load the test dataset from JSON file"""
    # Might have issues with file path
    try:
        with open('test_dataset.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("Warning: test_dataset.json not found")
        # TODO: fix this - try alternate path?
        return []

def preprocess_text(text, tokenizer):
    """Tokenize and prepare text for model input"""
    # Something seems wrong here - missing parameters?
    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt"
        # TODO: fix this - need max_length? truncation?
    )
    return inputs

def generate_summary(model, tokenizer, article_text):
    """Generate summary for a single article"""
    # Preprocess the input text
    input_ids = preprocess_text(article_text, tokenizer)
    
    # Generate summary with model
    # TODO: fix this - parameter names might be wrong
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=100,  # This might not work correctly
            temperature=0.7,
            do_sample=False
            # Something seems wrong here - missing min_length?
        )
    
    # Decode the generated summary
    # TODO: fix this - might have encoding issues
    summary = tokenizer.decode(outputs[0])
    
    return summary

def calculate_rouge_scores(predictions, references):
    """Calculate ROUGE scores for the predictions"""
    # TODO: fix this - rouge_score not imported
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        # Something seems wrong here - should we average?
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
        
        return avg_rouge1, avg_rougeL
    except ImportError:
        print("ERROR: rouge_score package not available")
        return 0.0, 0.0

def validate_summary_length(summary, tokenizer):
    """Check if summary meets length requirements (20-100 tokens)"""
    # TODO: fix this - not implemented properly
    tokens = tokenizer.encode(summary)
    length = len(tokens)
    # Should check if between 20-100 but logic is missing
    return True  # This always returns True!

def save_results(rouge1, rougeL, valid_count):
    """Save validation results to JSON file"""
    results = {
        "rouge1": rouge1,
        "rougeL": rougeL,
        "valid_summaries": valid_count
    }
    
    # TODO: fix this - output path might be wrong
    output_path = 'validation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")

def main():
    """Main execution function"""
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        return
    
    print("Loading test dataset...")
    test_data = load_test_data()
    
    if not test_data:
        print("No test data available. Exiting.")
        return
    
    print(f"Processing {len(test_data)} articles...")
    
    predictions = []
    references = []
    valid_summaries = 0
    
    for idx, item in enumerate(test_data):
        try:
            article = item['article']
            reference = item['reference_summary']
            
            # Generate summary
            summary = generate_summary(model, tokenizer, article)
            
            # TODO: fix this - validation not working correctly
            if validate_summary_length(summary, tokenizer):
                valid_summaries += 1
            
            predictions.append(summary)
            references.append(reference)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(test_data)} articles")
                
        except Exception as e:
            print(f"Error processing article {idx}: {e}")
            # Something seems wrong here - should we skip or fail?
            continue
    
    print("\nCalculating ROUGE scores...")
    rouge1, rougeL = calculate_rouge_scores(predictions, references)
    
    print(f"\nResults:")
    print(f"ROUGE-1 F1: {rouge1:.4f}")
    print(f"ROUGE-L F1: {rougeL:.4f}")
    print(f"Valid summaries: {valid_summaries}/{len(test_data)}")
    
    # Save results
    # TODO: fix this - might need different path
    save_results(rouge1, rougeL, valid_summaries)
    
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()