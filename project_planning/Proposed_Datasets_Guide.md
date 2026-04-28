# EE-559 Deep Learning Project: Proposed Datasets Guide

This document provides a comprehensive overview of the datasets proposed in the `Final_Project_Proposals.md` document. It includes descriptions, modalities, roles in the proposed projects, and instructions on how to access or download them.

---

## 1. Multimodal Datasets (Memes & Images)

### **Hateful Memes (Facebook / Meta AI)**

- **Role:** Primary training set for Proposals 1, 2, and 3.
- **Description:** A large-scale dataset of 10,000+ memes created by Facebook AI to challenge models on "compositional" hate (where image and text are individually benign but hateful together). It includes "benign confounders" designed to fool models.
- **Modality:** Image + Text.
- **Access:** Requires registration at [DrivenData's Hateful Memes Competition](https://www.drivendata.org/competitions/64/hateful-memes/).
- **Note:** You must manually agree to licensing terms before downloading.

### **MMHS150K**

- **Role:** Large-scale training/evaluation for Proposals 2 and 3.
- **Description:** 150,000 tweets containing images, manually annotated for hate speech. One of the largest multimodal datasets available.
- **Modality:** Image + Text.
- **Access:** [Gombru Repository](https://gombru.github.io/2019/10/09/MMHS/).
- **Paper:** [Exploring Hate Speech Detection in Multimodal Publications](https://arxiv.org/pdf/1910.03814.pdf).

### **HarMeme**

- **Role:** Training and target identification for Proposals 1 and 3.
- **Description:** Focuses on harmful memes and their targets, providing labels for the specific group being attacked.
- **Modality:** Image + Text.
- **Access:** [GitHub - di-dimitrov/harmeme](https://github.com/di-dimitrov/harmeme).
- **Paper:** [Detecting Harmful Memes and Their Targets](https://arxiv.org/abs/2110.00413).

### **MAMI (Multimedia Automatic Misogyny Identification)**

- **Role:** Gender-bias testing and misogyny detection in Proposals 1 and 2.
- **Description:** SemEval-2022 dataset containing 10,000 memes annotated for misogyny, shaming, stereotype, and violence.
- **Modality:** Image + Text.
- **Access:** Fill out the [MAMI Request Form](https://forms.gle/AGWMiGicBHiQx4q98) to receive the password-protected ZIP.
- **Paper:** [SemEval-2022 Task 5](https://aclanthology.org/2022.semeval-1.74/).

### **MultiOFF**

- **Role:** Offensive meme detection in Proposals 1 and 3.
- **Description:** A dataset for identifying offensive content in memes, covering a wide range of topics beyond just hate speech.
- **Modality:** Image + Text.
- **Access:** [Google Drive Folder](https://drive.google.com/drive/folders/1hKLOtpVmF45IoBmJPwojgq6XraLtHmV6).
- **Paper:** [Multimodal Meme Dataset (MultiOFF)](https://aclanthology.org/2020.trac-1.6.pdf).

---

## 2. Multimodal Datasets (Video & Audio)

### **HateMM**

- **Role:** Primary video training set for Proposal 4.
- **Description:** A multimodal dataset for hate video classification containing 1,000+ videos. Includes video-level labels and temporal rationales.
- **Modality:** Video + Audio + Text.
- **Access:** [Zenodo Repository](https://zenodo.org/records/7799469).
- **Paper:** [HateMM: A Multi-Modal Dataset for Hate Video Classification](https://arxiv.org/abs/2305.03915v1).

### **MultiHateClip**

- **Role:** Multilingual evaluation for Proposal 4.
- **Description:** A benchmark dataset for hateful video detection on YouTube and Bilibili, covering multiple languages.
- **Modality:** Video + Audio + Text.
- **Access:** [GitHub - social-ai-studio/multihateclip](https://github.com/social-ai-studio/multihateclip).
- **Paper:** [MultiHateClip: A Multilingual Benchmark Dataset](https://dl.acm.org/doi/pdf/10.1145/3664647.3681521).

### **HateClipSeg**

- **Role:** Temporal localization and fine-grained detection in Proposal 4.
- **Description:** Provides segment-level annotations (11,000+ segments) for offensive content in videos, allowing models to learn *when* the hate occurs.
- **Modality:** Video + Audio + Text.
- **Access:** [GitHub - Social-AI-Studio/HateClipSeg](https://github.com/Social-AI-Studio/HateClipSeg).
- **Paper:** [HateClipSeg: Segment-Level Annotated Dataset](https://arxiv.org/html/2508.01712).

### **MUStARD / MUStARD++**

- **Role:** Sarcasm supervision for Proposal 1.
- **Description:** Multimodal sarcasm detection dataset compiled from TV shows (Friends, Big Bang Theory). MUStARD++ adds emotion labels.
- **Modality:** Video + Audio + Text.
- **Access:** [MUStARD Repository](https://github.com/soujanyaporia/MUStARD) / [MUStARD++ Repository](https://github.com/cfiltnlp/MUStARD_Plus_Plus).
- **Paper:** [Towards Multimodal Sarcasm Detection](https://aclanthology.org/P19-1455/).

---

## 3. Text-Only Datasets

### **MetaHate**

- **Role:** Unified training resource for Proposal 5.
- **Description:** A dataset unifying multiple existing hate speech datasets (MetaHate) with 1.4 million entries, providing a standardized taxonomy.
- **Access:** [GitHub - palomapiot/metahate](https://github.com/palomapiot/metahate).
- **Paper:** [MetaHate: A Dataset for Unifying Efforts on Hate Speech Detection](https://ojs.aaai.org/index.php/ICWSM/article/view/31445/33605).

### **HateXplain**

- **Role:** Explainability training and bias auditing in Proposals 1, 4, and 5.
- **Description:** A benchmark dataset that includes not just labels but also "rationales" (tokens highlighting why a text is hateful) and target labels.
- **Access:** [GitHub - hate-alert/HateXplain](https://github.com/hate-alert/HateXplain).
- **Paper:** [HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection](https://ojs.aaai.org/index.php/AAAI/article/view/17745).

### **OLID (Offensive Language Identification Dataset)**

- **Role:** Policy-conditioned training in Proposal 5.
- **Description:** A 3-level hierarchy dataset: (A) Offensive vs. Non-offensive, (B) Targeted vs. Untargeted, (C) Target Type (Individual, Group, Other).
- **Access:** [GitHub Repository (via SemEval)](https://github.com/msang/olid).
- **Paper:** [Predicting the Type and Target of Offensive Posts](https://arxiv.org/pdf/1902.09666v2).

### **Davidson (Hate Speech and Offensive Language)**

- **Role:** Comparative training in Proposal 5.
- **Description:** Distinguishes between hate speech, offensive language (e.g., profanity), and neither.
- **Access:** [GitHub - t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language).
- **Paper:** [Automated Hate Speech Detection and the Problem of Offensive Language](https://ojs.aaai.org/index.php/ICWSM/article/view/14955/14805).

### **Civil Comments (Jigsaw)**

- **Role:** Toxicity classification and bias testing in Proposal 5.
- **Description:** Large dataset (2M+ comments) from Jigsaw/Google with toxicity scores and identity labels for bias analysis.
- **Access:** [HuggingFace - google/civil_comments](https://huggingface.co/datasets/google/civil_comments).
- **Paper:** [Nuanced Metrics for Measuring Unintended Bias](https://dl.acm.org/doi/10.1145/3308560.3317593).

### **Implicit Hate Dataset**

- **Role:** Latent hate detection in Proposals 3 and 5.
- **Description:** Focuses on "hidden" hate speech (incitement, dehumanization, stereotypes) that doesn't use slurs.
- **Access:** [GitHub - SALT-NLP/implicit-hate](https://github.com/SALT-NLP/implicit-hate).
- **Paper:** [Latent Hatred: A Benchmark for Understanding Implicit Hate Speech](https://aclanthology.org/2021.emnlp-main.29.pdf).

### **ETHOS**

- **Role:** Multi-label hate and target prediction in Proposals 1 and 5.
- **Description:** Multi-label dataset covering 8 dimensions (gender, race, sexual orientation, etc.).
- **Access:** [GitHub - Ethos-Hate-Speech-Dataset](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset).
- **Paper:** [ETHOS: an Online Hate Speech Detection Dataset](https://link.springer.com/epdf/10.1007/s40747-021-00608-2).

### **iSarcasm / News Headlines Sarcasm**

- **Role:** Sarcasm detection training for Proposal 1.
- **Description:** iSarcasm contains author-labeled sarcasm with explanations. News Headlines contains headlines from The Onion vs. HuffPost.
- **Access:** [iSarcasm GitHub](https://github.com/dmbavkar/iSarcasm) / [News Headlines (Kaggle)](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection/data).

---

## 4. Specialized Knowledge & Fairness Resources

### **ConceptNet**

- **Role:** Knowledge-augmented reasoning in Proposal 3.
- **Description:** A massive knowledge graph representing common sense relationships (e.g., "A bat is used for baseball").
- **Access:** [ConceptNet Website/API](https://conceptnet.io) or [Flat file download](https://github.com/commonsense/conceptnet5).

### **FairPrism**

- **Role:** Bias auditing framework for Proposal 2.
- **Description:** A 5,000-example dataset with detailed annotations for fairness-related harms.
- **Access:** Reach out to `fairprism@microsoft.com` or check the [GitHub Repository](https://github.com/microsoft/FairPrism).

### **Racial Stereotypes Dataset**

- **Role:** Racial bias testing in Proposal 2.
- **Description:** Multilingual dataset of racial stereotypes in social media threads.
- **Access:** Available upon request to the authors ([Paper link](https://aclanthology.org/2023.findings-eacl.51/)).

### **Hatemoji**

- **Role:** Emoji-based hate challenge for Proposal 3.
- **Description:** Test suite and adversarially-generated dataset for emoji-based hate.
- **Access:** [GitHub - HannahKirk/Hatemoji](https://github.com/HannahKirk/Hatemoji).

