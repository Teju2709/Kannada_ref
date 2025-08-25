#!/usr/bin/env python3
"""
Complete Kannada OCR Processing Pipeline
OCR → Post-processing → Gemini AI Refinement → Clean Editable Text
"""

import os
import sys
import argparse
import json
from pathlib import Path
# ...existing code...
import api  # This will set up the API key
import google.generativeai as genai

# Now you can use genai as needed
# ...existing code...
# Import your existing classes (assuming they're in the same directory)
try:
    from kannada_ocr import KannadaPDFOCR
    from kannada_postprocessor import KannadaPostProcessor
    from gemini_kannada_refiner import GeminiKannadaRefiner
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all three files are in the same directory:")
    print("- kannada_ocr.py")
    print("- kannada_postprocessor.py") 
    print("- gemini_kannada_refiner.py")
    sys.exit(1)

class CompleteKannadaPipeline:
    """
    Complete pipeline for processing old Kannada documents
    """
    
    def __init__(self, gemini_api_key: str, tesseract_path: str = None):
        """
        Initialize the complete processing pipeline
        
        Args:
            gemini_api_key: Google AI API key for Gemini
            tesseract_path: Path to tesseract executable (optional)
        """
        # Initialize all processors
        self.ocr_processor = KannadaPDFOCR(tesseract_path)
        self.post_processor = KannadaPostProcessor()
        self.gemini_refiner = GeminiKannadaRefiner(gemini_api_key)
        
        # Pipeline configuration
        self.config = {
            'ocr_dpi': 300,
            'enhancement_level': 'medium',  # light, medium, heavy
            'save_intermediate': True,
            'gemini_chunk_size': 1000,
            'max_retries': 3
        }
    
    def process_pdf_complete(self, pdf_path: str, output_base: str, 
                           document_context: str = None) -> dict:
        """
        Complete processing pipeline from PDF to refined text
        
        Args:
            pdf_path: Input PDF file path
            output_base: Base name for all output files
            document_context: Context about the document type
            
        Returns:
            Processing results dictionary
        """
        results = {
            'input_pdf': pdf_path,
            'output_base': output_base,
            'stages': {},
            'final_files': [],
            'success': False
        }
        
        try:
            print(f"\n{'='*60}")
            print(f"COMPLETE KANNADA DOCUMENT PROCESSING PIPELINE")
            print(f"{'='*60}")
            print(f"Input PDF: {pdf_path}")
            print(f"Output Base: {output_base}")
            if document_context:
                print(f"Context: {document_context}")
            print(f"{'='*60}\n")
            
            # Stage 1: OCR Processing
            print("STAGE 1: OCR PROCESSING")
            print("-" * 30)
            
            raw_text_file = f"{output_base}_raw_ocr.txt"
            ocr_success = self.ocr_processor.process_pdf(
                pdf_path, 
                raw_text_file,
                enhancement_level=self.config['enhancement_level'],
                save_enhanced_images=self.config['save_intermediate']
            )
            
            if not ocr_success:
                results['stages']['ocr'] = {'success': False, 'error': 'OCR processing failed'}
                return results
            
            # Read raw OCR output
            with open(raw_text_file, 'r', encoding='utf-8') as f:
                raw_ocr_text = f.read()
            
            results['stages']['ocr'] = {
                'success': True,
                'output_file': raw_text_file,
                'text_length': len(raw_ocr_text)
            }
            
            print(f"✓ OCR completed - {len(raw_ocr_text)} characters extracted")
            
            # Stage 2: Post-processing
            print("\nSTAGE 2: POST-PROCESSING")
            print("-" * 30)
            
            postprocess_base = f"{output_base}_postprocessed"
            postprocess_results = self.post_processor.process_full_pipeline(
                raw_ocr_text, 
                postprocess_base
            )
            
            # Read post-processed text
            postprocessed_file = f"{postprocess_base}.txt"
            with open(postprocessed_file, 'r', encoding='utf-8') as f:
                postprocessed_text = f.read()
            
            results['stages']['postprocessing'] = {
                'success': True,
                'output_file': postprocessed_file,
                'text_length': len(postprocessed_text),
                'corrections_made': postprocess_results['corrections_made'],
                'segments_detected': postprocess_results['segments_detected']
            }
            
            print(f"✓ Post-processing completed - {postprocess_results['corrections_made']} corrections made")
            
            # Stage 3: Gemini AI Refinement
            print("\nSTAGE 3: AI REFINEMENT WITH GEMINI")
            print("-" * 30)
            
            refined_base = f"{output_base}_refined"
            refinement_results = self.gemini_refiner.refine_full_document(
                postprocessed_file,
                refined_base,
                chunk_size=self.config['gemini_chunk_size'],
                context=document_context
            )
            
            if 'error' in refinement_results:
                results['stages']['refinement'] = {
                    'success': False, 
                    'error': refinement_results['error']
                }
                print(f"✗ AI refinement failed: {refinement_results['error']}")
                
                # Still consider pipeline successful if we have post-processed output
                results['success'] = True
                results['final_files'] = [postprocessed_file, f"{postprocess_base}.docx"]
                return results
            
            results['stages']['refinement'] = {
                'success': True,
                'output_files': refinement_results['output_files'],
                'original_length': refinement_results['original_length'],
                'refined_length': refinement_results['refined_length'],
                'improvement_ratio': refinement_results['improvement_ratio'],
                'processing_time': refinement_results['processing_time']
            }
            
            print(f"✓ AI refinement completed")
            print(f"  - Improvement ratio: {refinement_results['improvement_ratio']:.2f}x")
            print(f"  - Processing time: {refinement_results['processing_time']:.2f} seconds")
            
            # Stage 4: Final cleanup and summary
            print("\nSTAGE 4: FINAL PROCESSING")
            print("-" * 30)
            
            self._create_final_summary(results, output_base)
            
            results['success'] = True
            results['final_files'] = refinement_results['output_files']
            
            print(f"\n{'='*60}")
            print("PROCESSING COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            
            return results
            
        except Exception as e:
            print(f"\n✗ Pipeline failed with error: {e}")
            results['stages']['pipeline_error'] = {'error': str(e)}
            return results
    
    def _create_final_summary(self, results: dict, output_base: str):
        """
        Create a comprehensive summary of the entire processing pipeline
        
        Args:
            results: Processing results dictionary
            output_base: Base output filename
        """
        summary = {
            'pipeline_version': '1.0',
            'processing_timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
            'input_pdf': results['input_pdf'],
            'processing_stages': results['stages'],
            'success': results['success'],
            'final_output_files': results.get('final_files', []),
            'recommendations': []
        }
        
        # Add recommendations based on processing results
        if results['success']:
            summary['recommendations'].append("✓ Processing completed successfully")
            
            if 'refinement' in results['stages'] and results['stages']['refinement']['success']:
                improvement = results['stages']['refinement']['improvement_ratio']
                if improvement > 1.5:
                    summary['recommendations'].append("✓ Significant text improvement achieved through AI refinement")
                elif improvement > 1.2:
                    summary['recommendations'].append("✓ Good text improvement achieved")
                else:
                    summary['recommendations'].append("• Minor improvements made - original OCR was relatively clean")
            
            # Quality assessment
            ocr_length = results['stages']['ocr']['text_length']
            if 'refinement' in results['stages'] and results['stages']['refinement']['success']:
                final_length = results['stages']['refinement']['refined_length']
                if final_length > ocr_length * 1.3:
                    summary['recommendations'].append("✓ Substantial content recovery achieved")
                
            postprocess_corrections = results['stages']['postprocessing']['corrections_made']
            if postprocess_corrections > 20:
                summary['recommendations'].append("• Original OCR had many errors - refinement was essential")
            elif postprocess_corrections > 5:
                summary['recommendations'].append("• Moderate OCR errors corrected")
            
        else:
            summary['recommendations'].append("✗ Processing incomplete - check error messages")
        
        # Save summary
        summary_file = f"{output_base}_COMPLETE_SUMMARY.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Complete summary saved: {summary_file}")
    
    def batch_process_pdfs(self, pdf_directory: str, output_directory: str, 
                          document_context: str = None) -> dict:
        """
        Process multiple PDF files in batch
        
        Args:
            pdf_directory: Directory containing PDF files
            output_directory: Directory for output files
            document_context: Context for all documents
            
        Returns:
            Batch processing results
        """
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        
        if not pdf_files:
            return {'error': 'No PDF files found in directory'}
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING {len(pdf_files)} PDF FILES")
        print(f"{'='*60}")
        
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)
        
        batch_results = {
            'total_files': len(pdf_files),
            'successful': 0,
            'failed': 0,
            'results': {},
            'summary': {}
        }
        
        for i, pdf_file in enumerate(pdf_files):
            print(f"\n[{i+1}/{len(pdf_files)}] Processing: {pdf_file.name}")
            
            output_base = os.path.join(output_directory, pdf_file.stem)
            result = self.process_pdf_complete(str(pdf_file), output_base, document_context)
            
            batch_results['results'][pdf_file.name] = result
            
            if result['success']:
                batch_results['successful'] += 1
                print(f"✓ {pdf_file.name} processed successfully")
            else:
                batch_results['failed'] += 1
                print(f"✗ {pdf_file.name} processing failed")
        
        # Save batch summary
        batch_summary_file = os.path.join(output_directory, "BATCH_SUMMARY.json")
        with open(batch_summary_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETED")
        print(f"{'='*60}")
        print(f"Total files: {batch_results['total_files']}")
        print(f"Successful: {batch_results['successful']}")
        print(f"Failed: {batch_results['failed']}")
        print(f"Success rate: {batch_results['successful']/batch_results['total_files']*100:.1f}%")
        print(f"Batch summary: {batch_summary_file}")
        
        return batch_results
    
    def set_config(self, **kwargs):
        """
        Update pipeline configuration
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)
        print(f"Pipeline configuration updated: {kwargs}")

def main():
    """
    Main function for command line usage
    """
    parser = argparse.ArgumentParser(description='Complete Kannada OCR Processing Pipeline')
    parser.add_argument('input', help='Input PDF file or directory of PDF files')
    parser.add_argument('output', help='Output directory or base filename')
    from api import API_KEY
    parser.add_argument('--api-key', default=API_KEY, help='Google AI API key for Gemini')
    parser.add_argument('--context', help='Document context (e.g., "historical literature", "newspaper", "government document")')
    parser.add_argument('--tesseract-path', help='Path to tesseract executable')
    parser.add_argument('--enhancement', choices=['light', 'medium', 'heavy'], 
                       default='medium', help='OCR image enhancement level')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Text chunk size for Gemini processing')
    parser.add_argument('--batch', action='store_true', help='Process directory of PDFs in batch mode')
    parser.add_argument('--no-intermediate', action='store_true', help='Don\'t save intermediate files')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    try:
        pipeline = CompleteKannadaPipeline(args.api_key, args.tesseract_path)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return
    
    # Configure pipeline
    pipeline.set_config(
        enhancement_level=args.enhancement,
        gemini_chunk_size=args.chunk_size,
        save_intermediate=not args.no_intermediate
    )
    
    # Process files
    if args.batch:
        # Batch processing
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            return
        
        results = pipeline.batch_process_pdfs(args.input, args.output, args.context)
        
    else:
        # Single file processing
        if not os.path.isfile(args.input):
            print(f"Error: {args.input} is not a file")
            return
        
        results = pipeline.process_pdf_complete(args.input, args.output, args.context)
    
    # Final status
    if isinstance(results, dict) and results.get('success', False):
        print(f"\n🎉 Processing completed successfully!")
        if 'final_files' in results:
            print(f"📄 Final output files:")
            for file in results['final_files']:
                print(f"   - {file}")
    else:
        print(f"\n❌ Processing failed or incomplete")

# Example usage function
def example_usage():
    """
    Show example usage of the pipeline
    """
    print("""
KANNADA OCR PROCESSING PIPELINE - EXAMPLE USAGE
===============================================

1. SINGLE FILE PROCESSING:
   python integrated_pipeline.py input.pdf output_base --api-key YOUR_API_KEY

2. BATCH PROCESSING:
   python integrated_pipeline.py pdf_folder/ output_folder/ --api-key YOUR_API_KEY --batch

3. WITH CONTEXT:
   python integrated_pipeline.py input.pdf output_base --api-key YOUR_API_KEY --context "historical literature"

4. ADVANCED OPTIONS:
   python integrated_pipeline.py input.pdf output_base \\
       --api-key YOUR_API_KEY \\
       --enhancement heavy \\
       --chunk-size 800 \\
       --context "newspaper articles"

GETTING GOOGLE AI API KEY:
=========================
1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Use it with --api-key parameter

RECOMMENDED CONTEXTS:
====================
- "historical literature" - for old books, poems, stories
- "newspaper articles" - for newspaper content
- "government documents" - for official documents
- "religious texts" - for religious literature
- "educational content" - for textbooks, academic papers
- "personal letters" - for handwritten or typed letters

OUTPUT FILES:
============
The pipeline creates multiple output files:
- *_raw_ocr.txt - Raw OCR output
- *_postprocessed.txt/.docx - After basic post-processing
- *_refined.txt/.docx - After AI refinement (FINAL CLEAN TEXT)
- *_COMPLETE_SUMMARY.json - Processing summary and metrics

The *_refined.txt and *_refined.docx files contain the final,
clean, editable Kannada text ready for use.
    """)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        example_usage()
    else:
        main()