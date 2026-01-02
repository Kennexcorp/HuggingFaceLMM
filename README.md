# HuggingFace LMM Course

## Overview
This project demonstrates comprehensive exploration of HuggingFace transformers library, covering various natural language processing and computer vision tasks through hands-on experimentation with pre-trained models.

## What I've Learned

### 🤖 Pipeline API Fundamentals
- **Pipeline abstraction**: Learned how to use the high-level `pipeline()` function for quick model deployment without complex setup
- **Task-specific pipelines**: Explored different pipeline tasks for various NLP and computer vision applications
- **Model loading**: Understanding how to load pre-trained models from HuggingFace Hub using model identifiers

### 📝 Natural Language Processing Tasks

#### 1. Zero-Shot Classification
- **Model Used**: `facebook/bart-large-mnli`
- **Learning**: Classifying text into categories without training data using natural language inference
- **Application**: Automated content categorization and sentiment analysis without labeled datasets

#### 2. Text Generation
- **Model Used**: `gpt2`
- **Learning**: Generating coherent text continuation using transformer-based language models
- **Key Parameters**: `num_return_sequences`, `max_new_tokens` for controlling output

#### 3. Machine Translation
- **Model Used**: `google-t5/t5-base`
- **Learning**: Translation pipeline implementation for English to French
- **Application**: Cross-language communication and content localization

#### 4. Masked Language Modeling
- **Learning**: Filling in missing words in context using `fill-mask` pipeline
- **Application**: Text completion and cloze testing scenarios

#### 5. Named Entity Recognition (NER)
- **Learning**: Extracting entities (persons, locations, organizations) from text
- **Configuration**: `grouped_entities=True` for better entity grouping
- **Application**: Information extraction and document analysis

#### 6. Question Answering
- **Models Used**: 
  - Pipeline approach for quick implementation
  - `deepset/minilm-uncased-squad2` with `AutoTokenizer` and `TFAutoModelForQuestionAnswering`
- **Learning**: Converting PyTorch models to TensorFlow using `from_pt=True` parameter
- **Key Insight**: Simple parameter conversion reduces complex weight transformation to a single flag

#### 7. Text Summarization
- **Learning**: Creating concise summaries of longer texts using transformer models
- **Application**: Document processing and content summarization

### 🖼️ Computer Vision Tasks

#### 8. Image Classification
- **Model Used**: `google/vit-base-patch16-224` (Vision Transformer)
- **Learning**: Applying transformer architecture to image classification
- **Integration**: Working with PIL for image processing and matplotlib for visualization

#### 9. Audio Processing
- **Model Used**: `openai/whisper-large-v3`
- **Learning**: Automatic Speech Recognition (ASR) for audio transcription
- **Application**: Voice-to-text conversion and audio content analysis

## Technical Insights

### Model Loading Strategies
- **Pipeline approach**: Quick prototyping and experimentation
- **Direct model loading**: Using `AutoTokenizer` and specific model classes for fine-grained control
- **Cross-framework compatibility**: Successfully converting PyTorch models to TensorFlow format

### Best Practices Discovered
1. **Pipeline for simplicity**: Use `pipeline()` for quick experiments and prototypes
2. **Direct loading for customization**: Use `AutoTokenizer` and model classes when you need more control
3. **Framework conversion**: The `from_pt=True` parameter seamlessly converts PyTorch models to TensorFlow
4. **Resource management**: Understanding model sizes and memory requirements for different tasks

### Model Architecture Understanding
- **Transformer universality**: Same architecture applied across text, image, and audio domains
- **Pre-trained advantages**: Leveraging transfer learning for rapid deployment
- **Task specialization**: Different models optimized for specific tasks

## Files Structure
- `HuggingFace - Pipelines.ipynb`: Comprehensive pipeline demonstrations across multiple modalities
- `Hugging Face QnA.ipynb`: Focused exploration of question-answering systems
- `traffic.png`: Sample image for classification experiments
- `harvard.wav`: Audio sample for speech recognition testing
- `requirements.txt`: Project dependencies (access denied during README creation)

## Key Takeaways
1. **HuggingFace transformers provides unified API** across diverse AI tasks
2. **Transfer learning** enables quick deployment without extensive training
3. **Pipeline abstraction** significantly reduces complexity for experimentation
4. **Multi-modal capabilities** allow working with text, images, and audio using consistent interface
5. **Model conversion** between frameworks is simplified through built-in tools

## Future Learning Directions
- Fine-tuning pre-trained models on specific datasets
- Deploying models for production use
- Exploring smaller, more efficient models for resource-constrained environments
- Implementing custom training pipelines
- Model evaluation and performance optimization techniques

---

*This project represents foundational learning in modern AI/ML using state-of-the-art pre-trained models through the HuggingFace ecosystem.*
