# Deep Learning for Safer Online Spaces: Literature Review and Project Ideas

**EE-559 Deep Learning -- EPFL Master's Project Planning**
**Date: March 2026**

---

## Table of Contents

1. [Literature Review](#1-literature-review)
   - [1.1 Multimodal Hate Speech Detection](#11-multimodal-hate-speech-detection)
   - [1.2 Implicit and Coded Hate Speech](#12-implicit-and-coded-hate-speech)
   - [1.3 LLM-Based Approaches](#13-llm-based-approaches)
   - [1.4 Counter-Narrative Generation](#14-counter-narrative-generation)
   - [1.5 Explainability in Hate Speech Detection](#15-explainability-in-hate-speech-detection)
   - [1.6 Bias and Fairness](#16-bias-and-fairness)
   - [1.7 Temporal Dynamics and Evolving Hate Speech](#17-temporal-dynamics-and-evolving-hate-speech)
   - [1.8 Cross-Lingual and Cross-Domain Transfer](#18-cross-lingual-and-cross-domain-transfer)
   - [1.9 Gaming and Adolescent-Specific Hate Speech](#19-gaming-and-adolescent-specific-hate-speech)
2. [Project Ideas](#2-project-ideas)
3. [Ranked Top 5](#3-ranked-top-5)

---

## 1. Literature Review

### 1.1 Multimodal Hate Speech Detection

**State of the Art:**

Multimodal hate speech detection has advanced significantly in 2024-2025, moving beyond text-only classifiers to systems that jointly reason over text, images, audio, and video.

- **Memes:** CLIP-based methods dominate. MemeCLIP (2024) outperforms earlier BERT+ResNet fusion methods on the Facebook Hateful Memes benchmark. A CVPR 2025 workshop paper proposes leveraging frozen large multimodal models (LMMs) to generate knowledge about memes, then training lightweight classification heads on LMM-enriched representations, avoiding expensive fine-tuning. Definition-guided zero-shot prompting with VLMs is another emerging direction.
- **Video:** MM-HSD (ACM Multimedia 2025) integrates video frames, audio, and text transcripts via Cross-Modal Attention, achieving strong results on HateMM and MultiHateClip. The HCC1 system achieves SOTA M-F1 of 0.848 by combining HateXplain text features, CLIP visual features, and CLAP audio features via late fusion. TANDEM (2025) introduces a reinforcement learning strategy where vision-language and audio-language models optimize each other, achieving 0.73 F1 in target identification on HateMM (30% improvement over prior SOTA). ImpliHateVid (ACL 2025) introduces the first video dataset for implicit hate, with a two-stage contrastive learning framework.
- **Segment-level video analysis:** HateClipSeg (2025) provides segment-level annotations for 11,714 video segments across 5 offensive categories, supporting temporal hate localization. Research reveals that in HateMM, hateful videos average 2.56 minutes but hateful segments span only 1.71 minutes (33% is non-hateful content), highlighting the need for temporal granularity.

**Key Papers:**
- MM-HSD (ACM MM 2025): Multi-Modal Hate Speech Detection in Videos
- TANDEM (2025): Temporal-Aware Neural Detection for Multimodal Hate Speech
- ImpliHateVid (ACL 2025): Implicit hate speech detection in videos
- HateClipSeg (2025): Segment-level annotated dataset for fine-grained hate video detection
- MemeCLIP (2024): Leveraging CLIP for multimodal meme classification
- LMM-Generated Knowledge for Hateful Memes (CVPR 2025 Workshop)

**Gaps:**
1. **Sarcasm-aware multimodal detection:** Most video/meme detectors treat hate as binary/ternary; jointly detecting sarcasm, irony, and hate in multimodal content remains underexplored.
2. **Audio-specific hate signals:** Tone, prosody, and vocal emotion are underutilized; most systems only use audio transcripts, not acoustic features.
3. **Temporal localization:** Despite HateClipSeg, most methods still operate at video-level; fine-grained temporal grounding with explainable timestamps is nascent.
4. **Cross-modal reasoning for implicit hate in memes:** When image and text are individually benign but hateful in combination, current models still struggle significantly.

---

### 1.2 Implicit and Coded Hate Speech

**State of the Art:**

Implicit hate speech -- including dog whistles, coded language, sarcasm as a vehicle for hate, and context-dependent slurs -- remains one of the hardest challenges.

- **Attention amplification:** A 2025 EMNLP paper proposes amplifying attention mechanisms specifically for implicit hate detection, improving recall on subtle cases.
- **LLM-based relabeling and augmentation:** Adaptation pipelines using GPT-4o for relabeling and Llama-3 for paraphrastic augmentation improve F1 on implicit hate by +12.9 points without degrading explicit-hate performance.
- **Retrieval-augmented ICL (ARIIHA):** A novel framework that adaptively retrieves in-context demonstrations prioritizing similar target groups or high-similarity examples for few-shot implicit hate classification.
- **Contrastive learning for generalizability:** A COLING 2025 paper uses causality-guided contrastive learning to learn hate-speech representations that generalize across domains and datasets.
- **Dog whistle detection:** LLMs have been applied to word-sense disambiguation of dog whistles, creating the largest dataset (16,550 examples) of disambiguated dog whistle usage. However, reliable automated detection at scale remains an open problem.

**Key Papers:**
- Amplifying Attention for Implicit Hate Detection (EMNLP 2025)
- Selective Demonstration Retrieval for Implicit Hate Speech Detection (2025)
- Causality-guided Contrastive Learning for Generalizable Hate Speech Detection (COLING 2025)
- Silent Signals, Loud Impact: LLMs for Dog Whistle Disambiguation (2024)
- Specializing LLM Embeddings for Implicit Hate Speech Detection (2025)

**Gaps:**
1. **Commonsense and world-knowledge integration:** Many implicit hate utterances require political, cultural, or event-specific knowledge that text-only models lack.
2. **Multimodal implicit hate:** Almost no work on implicit hate in video or audio-visual content (ImpliHateVid is the sole pioneer).
3. **Dynamic dog whistles:** Coded language evolves rapidly; no system adapts to new dog whistles in real time.
4. **Cross-cultural implicit hate:** What is implicit hate in one culture may be explicit or benign in another; this mapping is understudied.

---

### 1.3 LLM-Based Approaches

**State of the Art:**

LLMs have emerged as powerful tools for hate speech detection, both as zero/few-shot classifiers and as components in larger systems.

- **Performance:** Open-source LLMs (LLaMA-3.1-8B, Mistral-7B, Qwen2.5-7B) outperform GPT-4o-mini on hate speech detection tasks. Fine-tuned LLaMA-3.1 consistently outperforms task-specific BERT models even with less training data.
- **Explainability via LLMs:** TARGE (2025) uses LLMs to generate explicit rationales by identifying critical textual features indicative of hate speech, producing more interpretable outputs than traditional classifiers.
- **Synthetic data generation:** LoRDS-GEN and similar methods use LLMs for demonstration-based augmentation, achieving 4-6% F1 improvements in low-resource settings. GPT-3-based augmentation avoids overfitting with up to 7x data increase and improves embedding coverage by 15%.
- **Limitations uncovered:** LLMs show over-sensitivity to group mentions (high false positive rates for benign content mentioning protected groups), are highly prompt-sensitive, underperform on code-mixed and transliterated text, and have poor confidence calibration.

**Key Papers:**
- Rethinking Hate Speech Detection: Can LLMs Replace Traditional Models? (2025)
- TARGE: LLM-Powered Explainable Hate Speech Detection (2025)
- LLM Synthetic Generation for Content Moderation (Computing, 2025)
- Hate Speech Detection using LLMs with Data Augmentation (2025)
- A Comprehensive Study on NLP Data Augmentation for Hate Speech (2024)

**Gaps:**
1. **LLM calibration for hate speech:** Over-sensitivity and confidence calibration remain unsolved.
2. **Efficient LLM deployment:** Most LLM approaches require large models; distillation or adapter-based methods for hate speech are underexplored.
3. **LLM + multimodal fusion:** Using LLMs as reasoning engines in multimodal pipelines (beyond just text) is nascent.
4. **Adversarial robustness of LLM-based detectors:** How easily can LLM-based detectors be fooled by adversarial perturbations?

---

### 1.4 Counter-Narrative Generation

**State of the Art:**

Counter-narrative (CN) generation has evolved from template-based to knowledge-grounded generative approaches.

- **Fact-based CN generation (ACM Web 2025):** Enhances counter narratives with relevant background knowledge from web search, producing non-aggressive, fact-based responses.
- **Retrieval-augmented approaches:** ReZG uses multi-dimensional hierarchical retrieval (stance, semantics, fitness) for zero-shot CN generation targeting unseen hate targets.
- **Knowledge-grounded systems:** PEACE 2.0 (2025) uses a knowledge base of 32,792 documents from UN Digital Library, Eur-Lex, and EU Fundamental Rights Agency to generate grounded counter-narratives and explanations.
- **Multilingual CN:** COLING 2025 hosted the First Workshop on Multilingual Counterspeech Generation, with shared tasks in 4 languages.
- **Stakeholder evaluation:** A 2025 study finds that most NLP-generated counter-speech lacks the contextual specificity and persuasiveness that stakeholders require in real-world moderation settings.

**Key Papers:**
- Fact-based Counter Narrative Generation (ACM Web 2025)
- PEACE 2.0: Grounded Explanations and Counter-Speech (2025)
- ReZG: Retrieval-Augmented Zero-Shot Counter Narrative Generation (Neurocomputing, 2024)
- NLP for Counterspeech: A Survey and How-To Guide (NAACL 2024)
- Contextualized Counterspeech: Strategies for Adaptation and Personalization (ACM Web 2025)

**Gaps:**
1. **Dialogue-level counter-narrative:** DIALOCONAN exists but generating multi-turn persuasive dialogue remains challenging; no system maintains coherent argumentation across turns.
2. **Multimodal counter-narratives:** All current CN systems are text-only; generating visual or video counter-narratives (counter-memes) is entirely unexplored.
3. **Personalization:** CNs are generic; adapting style, tone, and knowledge level to the target audience is underexplored.
4. **Evaluation gap:** No standardized evaluation framework exists; human evaluation is expensive and inconsistent.

---

### 1.5 Explainability in Hate Speech Detection

**State of the Art:**

A comprehensive 2026 review (WIREs) analyzed 63 studies and found that XAI in hate speech detection is dominated by post-hoc methods.

- **Dominant techniques:** SHAP, LIME, Integrated Gradients, and attention visualization remain the most used XAI methods.
- **LLM-generated rationales:** TARGE (2025) represents a shift toward using LLMs to generate free-text explanations rather than just highlighting tokens.
- **HateXplain benchmark:** Remains the primary benchmark (2021) with token-level rationale annotations, but the field has not produced updated successors with richer explanation types.
- **Stakeholder-centered evaluation:** Research finds that perceived fairness of moderation decisions depends not only on explanation clarity but also on whether users feel their perspectives are represented.

**Key Papers:**
- Explainable AI for Hate Speech Moderation: A Stakeholder-Centered Review (WIREs, 2026)
- TARGE: LLM-Powered Explainable Hate Speech Detection (2025)
- Decoding Fake News and Hate Speech: Survey of XAI Techniques (ACM Computing Surveys, 2025)
- HateXplain (AAAI 2021) -- foundational benchmark

**Gaps:**
1. **Multimodal explanations:** No system provides visual grounding (which part of the image is hateful?) + textual rationale jointly.
2. **Contrastive explanations:** "This is hateful because X, but would not be hateful if Y" -- this counterfactual style is almost absent.
3. **User-facing explanations:** Most XAI work targets researchers/developers, not end-users or content moderators.
4. **Faithfulness vs. plausibility tradeoff:** Even well-performing models score poorly on explanation faithfulness metrics.

---

### 1.6 Bias and Fairness

**State of the Art:**

- **The core problem:** Hate speech classifiers disproportionately flag content from African-American English (AAE) speakers and content mentioning protected groups, even when benign.
- **Geometric deep learning:** Graph-based approaches incorporating social network structure can achieve predictive equality (zero false positives among minority users) while maintaining accuracy.
- **Debiasing methods:** Adversarial debiasing, fairness constraints, and preprocessing interventions are the standard toolkit. A comprehensive taxonomy of bias mitigation methods was presented at WOAH 2025.
- **FairPrism:** A 5,000-example dataset with detailed annotations for fairness-related harms, accounting for annotator disagreement and context-dependent harms.

**Key Papers:**
- Tackling Racial Bias with Geometric Deep Learning (EPJ Data Science)
- Gender Bias Detection at Feature-Level (Neural Computing, 2024)
- A Comprehensive Taxonomy of Bias Mitigation Methods (WOAH 2025)
- FairPrism: Evaluating Fairness-Related Harms (ACL 2023)
- Investigating Annotator Bias in LLMs for Hate Speech Detection (2024)

**Gaps:**
1. **Bias in multimodal models:** Almost all bias research focuses on text; bias in meme/video hate detectors is unstudied.
2. **Intersectional bias:** Bias at the intersection of multiple protected attributes (e.g., Black + female + LGBTQ+) is not systematically evaluated.
3. **Bias from synthetic data:** LLM-generated training data may introduce new or amplify existing biases; this is not yet characterized.
4. **Fairness-explainability interaction:** How explanations interact with perceived fairness in moderation is underexplored.

---

### 1.7 Temporal Dynamics and Evolving Hate Speech

**State of the Art:**

- **Hatevolution (2025):** Demonstrates that static benchmarks are fundamentally limited because they cannot account for language change and evolving hate speech over time.
- **Continual learning:** Recognized as critical but not yet implemented in any deployed hate speech system. Future work should examine how model performance degrades as hate speech evolves.
- **Longitudinal analysis:** Most research uses short-term snapshots; longitudinal studies remain rare.

**Key Papers:**
- Hatevolution: What Static Benchmarks Don't Tell Us (2025)
- Moderating New Waves of Online Hate (2024)
- Tracing Online Hate Long-Term (2025)

**Gaps:**
1. **No continual learning system exists** for hate speech detection that adapts to new slang, dog whistles, and events without catastrophic forgetting.
2. **Temporal evaluation protocols** are absent; benchmarks do not test models on temporally shifted data.
3. **Event-triggered hate spikes** (e.g., political events, conflicts) require rapid model adaptation that current systems cannot provide.

---

### 1.8 Cross-Lingual and Cross-Domain Transfer

**State of the Art:**

- **Data-efficient methods (EMNLP 2025):** Retrieving as few as 200 labeled instances from a multilingual pool maintains strong cross-lingual detection across 8 languages.
- **Contrastive adversarial training (2025):** Combines supervised contrastive learning with adversarial training to transfer hate speech representations from high- to low-resource languages.
- **Metalinguist (2025):** Uses meta-learning for cross-lingual hate speech detection in underserved languages like Norwegian.
- **Domain-specific embeddings (2024):** First multilingual embedding model specifically designed for hate speech, enabling zero-shot cross-lingual evaluation.

**Key Papers:**
- Data-Efficient Cross-Lingual Hate Speech Detection via Nearest Neighbor Retrieval (EMNLP 2025)
- Enhancing Cross-Lingual Detection through Contrastive and Adversarial Learning (2025)
- Metalinguist: Cross-Lingual Meta-Learning for Hate Speech Detection (2025)
- Multilingual Hate Speech Detection and Counterspeech: Comprehensive Survey (2025)

**Gaps:**
1. **Cross-domain transfer** (e.g., social media to gaming, or memes to video) is barely explored compared to cross-lingual.
2. **Code-mixed and transliterated content** remains a failure mode for all models.
3. **Cultural adaptation:** Language transfer does not equal cultural transfer; hate speech definitions are culturally situated.

---

### 1.9 Gaming and Adolescent-Specific Hate Speech

**State of the Art:**

- **Scale of the problem:** 65% of parents notice hostile online behavior in gaming; one-third of adolescents in mobile games experience bullying.
- **Stream-based detection (2025):** Proposes incremental ML models with LLM feature engineering for real-time cyberbullying detection in streaming contexts.
- **ALONE dataset:** Specifically designed for adolescent hate speech, but underutilized in research.
- **Gaming datasets:** WoW/LoL Cyberbullying datasets exist but most research uses general social media data, not gaming-specific content.
- **Deep learning progress:** LSTM, Bi-LSTM, GRU, and transformer models show 5-12% accuracy improvements over traditional methods for cyberbullying detection.

**Key Papers:**
- Benchmarking LLMs for Cyberbullying Detection in YouTube Comments (2025)
- Stream-Based ML with LLMs for Cyberbullying Detection (2025)
- Deep Learning Models for Culturally Aware Cyberbullying Detection (2025)

**Gaps:**
1. **Gaming-specific language models:** No models are pre-trained or fine-tuned on gaming discourse (Twitch chat, Discord, in-game chat).
2. **Real-time multimodal gaming detection:** Voice chat + text chat + gameplay context fusion is unexplored.
3. **Age-appropriate moderation:** Distinguishing adolescent banter from genuine cyberbullying requires context current models lack.
4. **Longitudinal adolescent studies:** Tracking how exposure affects individuals over time and adapting models accordingly.

---

## 2. Project Ideas

### Idea 1: Temporal Multimodal Hate Speech Localization in Videos with Explainable Grounding

**Problem Statement:** Current video-level hate speech classifiers label entire videos as hateful/non-hateful, but hateful content often spans only a fraction of the video. Content moderators need precise temporal localization with multi-modal explanations.

**Key Innovation / Gap Filled:** Combines temporal hate speech localization with multimodal explanations (which modality -- text, audio tone, visual -- triggers hate at each timestamp). No existing system provides both temporal grounding AND multimodal rationales simultaneously.

**Proposed Approach:**
- Architecture: Multi-stream encoder (CLIP for frames, Whisper for audio/ASR, CLAP for audio emotion) with temporal attention pooling and a segment proposal network (inspired by temporal action localization).
- Training: Two-stage training -- (1) pre-train modality encoders with contrastive loss on HateClipSeg segment labels, (2) train temporal proposal network with cross-modal attention and auxiliary rationale prediction heads.
- Explainability: Per-segment attention scores decomposed by modality, plus LLM-generated natural language rationale for each detected hateful segment.

**Datasets:** HateClipSeg (segment-level annotations), HateMM (video-level with temporal rationales), MultiHateClip.

**Publication Potential:** High. Temporal localization with multimodal explanations is a novel combination. Suitable for ACM Multimedia, EMNLP, or a specialized workshop (WOAH).

**Challenges:** HateClipSeg segment annotations may not perfectly align with modality-specific hate cues. Computational cost of multi-stream processing. Evaluation of explanation quality requires human study.

---

### Idea 2: Continual Learning for Evolving Hate Speech with Event-Triggered Adaptation

**Problem Statement:** Hate speech evolves as new slang, dog whistles, and culturally-triggered phrases emerge (e.g., after political events or conflicts). Static models degrade over time, but no continual learning system exists for hate speech.

**Key Innovation / Gap Filled:** First continual learning framework specifically designed for hate speech detection, incorporating event-triggered adaptation and resistance to catastrophic forgetting. Directly addresses the gap identified by Hatevolution (2025).

**Proposed Approach:**
- Architecture: Transformer encoder (RoBERTa or DeBERTa) with Elastic Weight Consolidation (EWC) or progressive neural networks to prevent forgetting.
- Event detection module: Monitor keyword trends and semantic shift signals to trigger model updates when new hate waves are detected.
- Training protocol: Sequential fine-tuning on temporally ordered data splits from multiple datasets, measuring backward transfer (forgetting) and forward transfer (generalization).
- Evaluation: Novel temporal evaluation protocol -- train on pre-2023 data, evaluate on 2023-2024 data, then adapt and evaluate on 2024-2025 data.

**Datasets:** OLID, Davidson (older), TOXIGEN (newer), Civil Comments (with timestamps), HateXplain. Supplement with keyword-dated web scrapes (ethically sourced, public data).

**Publication Potential:** High. Continual learning for hate speech is explicitly identified as a gap by multiple 2025 survey papers. Novel evaluation protocol adds contribution.

**Challenges:** Constructing temporally ordered training splits with reliable timestamps. Defining "event triggers" systematically. Balancing adaptation speed vs. stability.

---

### Idea 3: Cross-Modal Implicit Hate Detection in Memes via Commonsense-Augmented Reasoning

**Problem Statement:** Hateful memes are particularly dangerous when image and text are individually benign but hateful in combination (e.g., a photo of a smiling person with sarcastic text implying a stereotype). Current models fail on these compositional cases.

**Key Innovation / Gap Filled:** Injects external commonsense knowledge (from knowledge graphs or LLMs) into the cross-modal reasoning pipeline to decode implicit hate that requires world knowledge. Addresses the intersection of two gaps: commonsense integration + multimodal implicit hate.

**Proposed Approach:**
- Architecture: Frozen CLIP encoder + commonsense retrieval module (from ConceptNet or LLM-generated commonsense triples) + cross-attention fusion + classification head.
- Pipeline: (1) Extract CLIP embeddings for image and text, (2) generate commonsense inferences about the meme (e.g., "person shown is associated with group X; text implies Y about group X"), (3) fuse original embeddings with commonsense embeddings via cross-attention, (4) classify.
- Training: Fine-tune fusion layers and classification head on Hateful Memes (Facebook) and HarMeme. Use LLM-generated commonsense triples as auxiliary features.

**Datasets:** Hateful Memes (Facebook), HarMeme, MMHS150K, Hatemoji. ConceptNet/LLM for commonsense.

**Publication Potential:** High. The CVPR 2025 workshop paper on LMM-generated knowledge shows this direction is hot, but explicit commonsense graph integration is novel.

**Challenges:** Quality of commonsense retrieval (noisy or irrelevant triples). Meme humor and cultural references are hard to formalize. Annotation subjectivity.

---

### Idea 4: Fairness-Aware Multimodal Hate Speech Detection with Bias Auditing

**Problem Statement:** Bias in hate speech classifiers is well-documented for text, but entirely unstudied for multimodal (meme/video) classifiers. A multimodal system that detects hate in images of Black people at higher rates than comparable images of White people would be deeply harmful.

**Key Innovation / Gap Filled:** First systematic bias audit of multimodal hate speech detectors, plus a debiasing method that operates across modalities. Addresses the gap of bias in multimodal models.

**Proposed Approach:**
- Bias audit: Evaluate SOTA multimodal hate detectors (MemeCLIP, Pro-Cap, CLIP-based classifiers) on FairPrism-style metrics across demographic groups. Create a counterfactual meme test set by swapping racial/gender/religious identities in images while keeping text constant.
- Debiasing: Apply adversarial debiasing to the multimodal fusion layer, training the classifier to be invariant to protected attributes while maintaining hate detection accuracy.
- Architecture: CLIP backbone + adversarial debiasing head at fusion layer + fairness-constrained loss function.

**Datasets:** Hateful Memes (Facebook), MMHS150K, FairPrism (for evaluation framework), Racial Stereotypes dataset. Create counterfactual test set via controlled image generation/swapping.

**Publication Potential:** Very high. First paper on multimodal hate speech bias; aligns perfectly with the "ML for good" emphasis and fairness community (FAccT, AIES).

**Challenges:** Creating high-quality counterfactual memes at scale. Defining and measuring bias in visual content. Disentangling legitimate contextual use from biased classification.

---

### Idea 5: Sarcasm-Aware Hate Speech Detection via Multimodal Sentiment-Hate Disentanglement

**Problem Statement:** Sarcastic hate speech is routinely misclassified by current detectors -- either flagged as benign (because surface sentiment is positive) or, conversely, benign sarcasm is flagged as hateful. No system jointly models sarcasm and hate in a multimodal setting.

**Key Innovation / Gap Filled:** A multi-task architecture that disentangles sarcasm detection from hate detection in multimodal content, learning separate but interacting representations for sentiment, sarcasm, and hate.

**Proposed Approach:**
- Architecture: Shared multimodal encoder (CLIP for image+text, or video encoder for MUStARD++) with three task-specific heads: (1) sarcasm detection, (2) sentiment classification, (3) hate speech classification. Use gradient reversal or adversarial training to ensure the hate head is not confused by sarcasm signals.
- Multi-task training: Joint training on sarcasm datasets (iSarcasm, MUStARD++) and hate speech datasets (Hateful Memes, HatEval) with a shared backbone, using task-weighted losses.
- Key insight: The model should learn that "sarcastic + targets protected group = potentially hateful sarcasm" vs. "sarcastic + targets self/general = likely benign."

**Datasets:** iSarcasm, MUStARD/MUStARD++ (multimodal sarcasm), Hateful Memes, HatEval, News Headlines Sarcasm, ETHOS.

**Publication Potential:** High. Sarcasm-hate interaction is frequently cited as a challenge but no dedicated solution exists. Multi-task multimodal approach is novel.

**Challenges:** Domain gap between sarcasm and hate speech datasets. Annotation quality for sarcastic hate is inherently subjective. Balancing task losses.

---

### Idea 6: Knowledge-Grounded Counter-Meme Generation

**Problem Statement:** Counter-narrative generation is exclusively text-based. In a visual social media landscape dominated by memes, an effective counter-narrative might itself be a counter-meme -- an image-text combination that debunks the hateful message.

**Key Innovation / Gap Filled:** First system to generate visual+textual counter-narratives (counter-memes) to combat hateful memes. Entirely novel modality for counter-speech.

**Proposed Approach:**
- Architecture: Hate meme encoder (CLIP) -> Knowledge retrieval module (searches factual knowledge base) -> Counter-text generator (fine-tuned LLM, e.g., Llama-3-8B) -> Counter-image selector/generator (retrieve from safe image bank or generate via controlled diffusion model).
- Training: Fine-tune the text generator on CONAN/Knowledge-grounded CONAN for counter-narrative text. Train the image selection module to choose images that visually contradict the hateful message while maintaining factual grounding.
- Evaluation: Human evaluation of counter-meme effectiveness (persuasiveness, factual accuracy, non-offensiveness) + automated metrics (BLEU, BERTScore for text; CLIP similarity for image relevance).

**Datasets:** CONAN, Knowledge-grounded CONAN, Hateful Memes (as input), MMHS150K. Open image databases for counter-image retrieval.

**Publication Potential:** Very high. Counter-meme generation is completely unexplored. Would be a "first of its kind" contribution.

**Challenges:** Generating appropriate images without introducing new harms. Evaluation is inherently subjective. Risk of the model producing offensive counter-content. Ethical considerations around automated counter-meme creation.

---

### Idea 7: Gaming Voice Chat Hate Speech Detection via Audio-Text Fusion

**Problem Statement:** Hate speech in gaming environments is predominantly delivered through voice chat, a modality that is severely underexplored. Existing text-based systems cannot process real-time audio with gaming-specific slang, code-switching, and emotional context.

**Key Innovation / Gap Filled:** First multimodal hate speech detector specifically designed for gaming voice chat, combining speech recognition, acoustic emotion features, and gaming-context understanding.

**Proposed Approach:**
- Architecture: Whisper (ASR) -> text encoder (fine-tuned on gaming slang) + audio emotion encoder (wav2vec 2.0 / HuBERT fine-tuned on paralinguistic features) -> fusion layer -> classification head.
- Gaming-specific adaptation: Fine-tune text encoder on WoW/LoL Cyberbullying datasets with gaming lexicon. Use SWAD dataset for additional domain-specific examples.
- Prosodic features: Extract pitch, intensity, speaking rate to detect aggressive tone even when words are individually benign.

**Datasets:** WoW/LoL Cyberbullying, SWAD, ALONE (adolescents), HateComments. May need to create a small gaming voice chat dataset (ethically sourced from consenting participants or public Twitch VODs with voice).

**Publication Potential:** Medium-high. Gaming + voice modality is underexplored. However, dataset creation may be needed, which limits scope.

**Challenges:** Lack of large-scale gaming voice chat datasets. Background noise, overlapping speakers, and low-quality audio. Ethical issues with recording/processing voice data.

---

### Idea 8: Distilling LLM Hate Speech Reasoning into Efficient Student Models

**Problem Statement:** LLMs achieve strong hate speech detection but are too expensive for real-time deployment at scale. Small models (BERT-size) are efficient but lack reasoning capability. No systematic distillation framework exists for hate speech.

**Key Innovation / Gap Filled:** A knowledge distillation framework that transfers LLM hate speech reasoning (including rationales and confidence calibration) into small, deployable models. Addresses the efficiency and calibration gaps.

**Proposed Approach:**
- Teacher: LLaMA-3.1-8B or Mistral-7B fine-tuned on hate speech with chain-of-thought rationale generation (using HateXplain rationales as supervision).
- Student: DeBERTa-base or DistilBERT, trained with (1) soft label distillation from teacher predictions, (2) rationale distillation (student learns to attend to the same tokens the teacher highlights), (3) calibration-aware loss to inherit teacher's confidence distribution.
- Evaluation: Compare student vs. teacher on accuracy, calibration (ECE), fairness (equalized odds), and explainability (rationale plausibility).

**Datasets:** HateXplain (rationales), OLID, Davidson, TOXIGEN, Civil Comments.

**Publication Potential:** Medium-high. Knowledge distillation is well-studied generally, but rationale + calibration distillation for hate speech is novel.

**Challenges:** Teacher quality bottleneck (LLM rationales are imperfect). Distillation of reasoning is harder than label distillation. Evaluating whether rationale transfer actually helps.

---

### Idea 9: Cross-Domain Hate Speech Transfer: From Social Media to Gaming

**Problem Statement:** Hate speech detectors trained on social media (Twitter/X, Reddit) fail when deployed in gaming (Twitch, Discord, in-game chat) due to massive domain shift in vocabulary, style, and context. Cross-domain transfer for hate speech is underexplored compared to cross-lingual.

**Key Innovation / Gap Filled:** First systematic study of cross-domain hate speech transfer between social media and gaming, with domain adaptation techniques. Addresses the gap explicitly: cross-domain transfer is "barely explored" in the literature.

**Proposed Approach:**
- Architecture: Shared transformer encoder with domain-adversarial neural network (DANN) training -- the encoder learns domain-invariant hate representations while a domain classifier is fooled.
- Domain adaptation: (1) Unsupervised domain adaptation from labeled social media (OLID, Davidson) to unlabeled gaming data (WoW/LoL), (2) Few-shot adaptation with a small labeled gaming set.
- Analysis: Characterize the domain gap (vocabulary analysis, hate type distribution, formality, code-switching) and identify which hate categories transfer well vs. poorly.

**Datasets:** Source: OLID, Davidson, TOXIGEN, HatEval. Target: WoW/LoL Cyberbullying, SWAD. Analysis: MetaHate for unified hate speech taxonomy.

**Publication Potential:** Medium-high. Cross-domain is an identified gap. The social-media-to-gaming direction is timely and practical.

**Challenges:** Gaming datasets are relatively small. Domain gap may be too large for simple adaptation. Need to account for gaming-specific phenomena (trash talk vs. hate speech).

---

### Idea 10: Hate Speech Detection with Annotator Disagreement Modeling

**Problem Statement:** Hate speech annotation is inherently subjective -- annotators from different backgrounds disagree significantly, especially on implicit, sarcastic, or borderline content. Standard practice aggregates labels (majority vote), discarding valuable disagreement signal. Models trained on aggregated labels learn a false consensus.

**Key Innovation / Gap Filled:** A hate speech detection model that explicitly models and preserves annotator disagreement, producing calibrated predictions with uncertainty estimates that reflect genuine ambiguity rather than forcing binary decisions.

**Proposed Approach:**
- Architecture: Transformer encoder + multiple prediction heads (one per annotator group or perspective), or a single head with a distributional output (predicting the full annotation distribution rather than a single label).
- Training: Use datasets with per-annotator labels (HateXplain has 3 annotators; Civil Comments has many). Train with a distributional loss (KL divergence to the empirical annotation distribution) rather than cross-entropy to a majority label.
- Uncertainty: Produce calibrated uncertainty estimates; high-disagreement cases are flagged for human review rather than auto-decided.
- Fairness connection: Analyze whether disagreement patterns correlate with annotator demographics and protected group mentions.

**Datasets:** HateXplain (per-annotator labels + rationales), Civil Comments, ETHOS (multi-label), FairPrism (detailed annotations with disagreement).

**Publication Potential:** High. Annotator disagreement modeling is a growing theme in NLP (Learning with Disagreement workshop), but application to hate speech with fairness analysis is novel.

**Challenges:** Datasets with per-annotator labels are rare and small. Defining "annotator groups" without stereotyping. Evaluation metrics for distributional predictions.

---

## 3. Ranked Top 5

The following ranking considers four criteria, each scored 1-5:
- **Innovation** (I): How novel is the contribution? Does it fill a clear gap?
- **Feasibility** (F): Can a Master's team complete this in ~2 months with available datasets and GPU cluster?
- **Publication Potential** (P): How likely is this to be accepted at a top venue?
- **Course Alignment** (C): Does it demonstrate DL technical depth and align with "ML for good"?

| Rank | Idea | I | F | P | C | Total |
|------|------|---|---|---|---|-------|
| **1** | **Idea 4: Fairness-Aware Multimodal Hate Speech Detection with Bias Auditing** | 5 | 4 | 5 | 5 | **19** |
| **2** | **Idea 5: Sarcasm-Aware Hate Speech Detection via Multimodal Sentiment-Hate Disentanglement** | 5 | 4 | 4 | 5 | **18** |
| **3** | **Idea 3: Cross-Modal Implicit Hate Detection via Commonsense-Augmented Reasoning** | 5 | 4 | 5 | 4 | **18** |
| **4** | **Idea 2: Continual Learning for Evolving Hate Speech** | 5 | 3 | 5 | 4 | **17** |
| **5** | **Idea 10: Hate Speech Detection with Annotator Disagreement Modeling** | 4 | 5 | 4 | 4 | **17** |

### Detailed Justification

**Rank 1 -- Fairness-Aware Multimodal Hate Speech Detection (Idea 4)**
- This is the strongest combination of novelty, impact, and alignment. Bias in multimodal hate detectors is *completely unstudied* -- the project would be genuinely first-of-its-kind. The methodology (counterfactual meme generation + adversarial debiasing at fusion layer) is technically deep. It directly serves the "ML for good" mission. Feasibility is good because it builds on existing CLIP-based architectures and the Hateful Memes dataset. The counterfactual test set creation is the main challenge but can be scoped to a manageable size.

**Rank 2 -- Sarcasm-Aware Hate Speech Detection (Idea 5)**
- Sarcasm-hate interaction is universally cited as a hard problem but nobody has built a dedicated solution. The multi-task disentanglement architecture is technically interesting and demonstrates DL depth (gradient reversal, multi-head, cross-dataset training). MUStARD++ provides ready-made multimodal sarcasm data. Feasibility is strong because the core components (CLIP, multi-task heads) are well-understood. The main risk is the domain gap between sarcasm and hate datasets.

**Rank 3 -- Commonsense-Augmented Implicit Hate in Memes (Idea 3)**
- Directly extends the CVPR 2025 workshop direction (LMM-generated knowledge for memes) with a more structured commonsense injection approach. High publication potential because the Hateful Memes challenge remains open and commonsense integration is the logical next step. Feasibility relies on CLIP (frozen) + lightweight fusion, which is GPU-friendly. The risk is that commonsense retrieval quality may be noisy.

**Rank 4 -- Continual Learning for Evolving Hate Speech (Idea 2)**
- Fills the most explicitly called-out gap in recent surveys (Hatevolution 2025). The contribution is both methodological (CL framework) and evaluative (temporal protocol). Lower feasibility score because constructing temporally ordered splits and event triggers requires careful data engineering. Publication potential is very high at venues like EMNLP, ACL, or NAACL.

**Rank 5 -- Annotator Disagreement Modeling (Idea 10)**
- Highest feasibility because it requires no new data collection and uses existing per-annotator labels. The distributional loss approach is elegant and technically sound. Slightly lower innovation score because disagreement modeling is studied in other NLP tasks, but application to hate speech with fairness analysis adds novelty. Well-suited to the course timeline.

---

## Sources

### Multimodal Hate Speech Detection
- [A comprehensive framework for multi-modal hate speech detection (Scientific Reports, 2025)](https://www.nature.com/articles/s41598-025-94069-z)
- [MM-HSD: Multi-Modal Hate Speech Detection in Videos (ACM MM 2025)](https://dl.acm.org/doi/10.1145/3746027.3754558)
- [TANDEM: Temporal-Aware Neural Detection for Multimodal Hate Speech](https://arxiv.org/html/2601.11178)
- [ImpliHateVid: Implicit Hate Speech Detection in Videos (ACL 2025)](https://aclanthology.org/2025.acl-long.842/)
- [HateClipSeg: Segment-Level Annotated Dataset](https://arxiv.org/html/2508.01712)
- [MemeCLIP: Leveraging CLIP for Meme Classification](https://arxiv.org/html/2409.14703v1)
- [Improving Hateful Meme Detection with LMM-Generated Knowledge (CVPR 2025)](https://arxiv.org/html/2504.09914v1)
- [Towards a Robust Framework for Multimodal Hate Detection: Video vs. Image (ACM Web 2025)](https://doi.org/10.1145/3701716.3718382)

### Implicit and Coded Hate Speech
- [Amplifying Attention for Implicit Hate Detection (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1469.pdf)
- [Selective Demonstration Retrieval for Implicit Hate Speech Detection](https://arxiv.org/html/2504.12082)
- [Causality-guided Contrastive Learning for Generalizable Hate Speech Detection (COLING 2025)](https://aclanthology.org/2025.coling-main.593.pdf)
- [Specializing LLM Embeddings for Implicit Hate Speech Detection](https://dl.acm.org/doi/10.1145/3746275.3762209)
- [Silent Signals, Loud Impact: LLMs for Dog Whistle Disambiguation](https://arxiv.org/html/2406.06840v1)

### LLM-Based Approaches
- [Rethinking Hate Speech Detection: Can LLMs Replace Traditional Models?](https://arxiv.org/html/2506.12744v1)
- [Beyond Traditional Classifiers: Evaluating LLMs for Hate Speech Detection](https://www.mdpi.com/2079-3197/13/8/196)
- [TARGE: LLM-Powered Explainable Hate Speech Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC12192871/)
- [LLM Synthetic Generation for Content Moderation (Computing, 2025)](https://link.springer.com/article/10.1007/s00607-025-01518-8)
- [Hate Speech Detection using LLMs with Data Augmentation](https://arxiv.org/html/2603.04698)

### Counter-Narrative Generation
- [Fact-based Counter Narrative Generation (ACM Web 2025)](https://dl.acm.org/doi/10.1145/3696410.3714718)
- [PEACE 2.0: Grounded Explanations and Counter-Speech](https://arxiv.org/html/2602.17467)
- [ReZG: Retrieval-Augmented Zero-Shot Counter Narrative Generation](https://arxiv.org/abs/2310.05650)
- [NLP for Counterspeech: A Survey (NAACL 2024)](https://aclanthology.org/2024.findings-naacl.221.pdf)
- [Contextualized Counterspeech: Adaptation and Personalization (ACM Web 2025)](https://dl.acm.org/doi/10.1145/3696410.3714507)

### Explainability
- [Explainable AI for Hate Speech Moderation: Stakeholder-Centered Review (WIREs, 2026)](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.70076)
- [Decoding Fake News and Hate Speech: Survey of XAI Techniques (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/3711123)
- [HateXplain: Benchmark for Explainable Hate Speech Detection (AAAI 2021)](https://arxiv.org/abs/2012.10289)

### Bias and Fairness
- [Tackling Racial Bias with Geometric Deep Learning (EPJ Data Science)](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-022-00319-9)
- [Gender Bias Detection at Feature-Level (Neural Computing, 2024)](https://link.springer.com/article/10.1007/s00521-024-10841-8)
- [A Comprehensive Taxonomy of Bias Mitigation Methods (WOAH 2025)](https://aclanthology.org/2025.woah-1.1.pdf)
- [FairPrism: Evaluating Fairness-Related Harms (ACL 2023)](https://aclanthology.org/2023.acl-long.343/)

### Temporal Dynamics
- [Hatevolution: What Static Benchmarks Don't Tell Us (2025)](https://arxiv.org/html/2506.12148)
- [Revealing Temporal Label Noise in Multimodal Hateful Video Classification](https://arxiv.org/abs/2508.04900)
- [A Comprehensive Review on Automatic Hate Speech Detection in the Age of the Transformer](https://link.springer.com/article/10.1007/s13278-024-01361-3)

### Cross-Lingual Transfer
- [Data-Efficient Cross-Lingual Detection via Nearest Neighbor Retrieval (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1507/)
- [Enhancing Cross-Lingual Detection through Contrastive and Adversarial Learning](https://www.sciencedirect.com/science/article/abs/pii/S0952197625002969)
- [Metalinguist: Cross-Lingual Meta-Learning (2025)](https://link.springer.com/article/10.1007/s40747-025-01808-w)
- [Multilingual Hate Speech Detection and Counterspeech: Survey (2025)](https://arxiv.org/html/2603.19279v1)

### Gaming and Cyberbullying
- [Benchmarking LLMs for Cyberbullying Detection in YouTube Comments (2025)](https://arxiv.org/html/2505.18927v1)
- [Stream-Based ML with LLMs for Cyberbullying Detection (2025)](https://arxiv.org/html/2505.03746v1)
- [Deep Learning for Culturally Aware Cyberbullying Detection (2025)](https://link.springer.com/article/10.1007/s44163-025-00577-2)

### Data Augmentation
- [A Comprehensive Study on NLP Data Augmentation for Hate Speech Detection (2024)](https://arxiv.org/abs/2404.00303)
- [Low-Resource Dataset Synthetic Generation for Hate Speech Detection (WISE 2024)](https://link.springer.com/chapter/10.1007/978-981-96-1483-7_6)
