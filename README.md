# Revealing Cross-Lingual Bias in Synthetic Speech Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Revealing Cross-Lingual Bias in Synthetic Speech Detection under Controlled Conditions"**

Victor Moreno¹, João Lima¹, Flávio Simões², Ricardo Violato², Mário Uliani Neto², Fernando Runstein², Paula Costa¹

¹Universidade Estadual de Campinas (UNICAMP), Brazil  
²CPQD, Brazil

## 📋 Abstract

This work investigates whether language identity influences the detectability of synthetic speech in state-of-the-art countermeasure systems. We train a detector on English-only data (ASVspoof 5) and evaluate it under controlled conditions using spoofed samples in ten languages synthesized by Meta's MMS TTS system. Despite uniform synthesis settings, we observe significant language-dependent disparities in detection performance, revealing systematic bias in cross-lingual generalization.

## 🎯 Key Findings

- **Language-dependent detection bias confirmed**: Detection performance varies significantly across languages despite identical TTS synthesis conditions
- **Counter-intuitive results**: English (training language) does not achieve the best detection scores
- **Extreme performance gaps**: Romanian (99% mean score) vs Ukrainian (12% mean score)
- **Statistical significance**: Mann-Whitney U tests confirm systematic differences (p < 0.001 for most pairs)

## 📊 Datasets

### Training Dataset
- **ASVspoof 5 Track 01**: English-only corpus
  - 145,000+ utterances (balanced bonafide/spoof)
  - Various TTS and VC systems
  - Used for monolingual training

### Evaluation Dataset  
- **MLAAD MMS Subset**: Controlled multilingual evaluation
  - 10 languages: Finnish, German, Russian, Swahili, Ukrainian, English, French, Dutch, Hungarian, Romanian
  - 1,000 spoofed utterances per language
  - All generated with identical MMS TTS settings (VITS-based, 100k steps, same architecture)
  - Enables isolation of language as the only variable

## 🛠 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone repository
git clone https://github.com/victorgmoreno/crosslingual_bias_audiodeepfake.git
cd crosslingual_bias_audiodeepfake

# Create environment
conda create -n crosslingual python=3.8
conda activate crosslingual

# Install PyTorch (adjust CUDA version)
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```txt
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
transformers>=4.20.0
fairseq>=0.12.0
```

## 🏗 Model Architecture

### AASIST + wav2vec2 Pipeline

```
Raw Audio (16kHz)
    ↓
wav2vec2 XLS-R 300M (Front-end)
    ↓
2D Self-Attentive Pooling
    ↓
AASIST (Back-end)
    - Heterogeneous Graph Attention
    - Spectro-temporal modeling
    ↓
CM Score [0,1]
```

- **Front-end**: wav2vec2 XLS-R 300M (pretrained on 436K hours multilingual speech)
- **Back-end**: AASIST with graph attention networks
- **Output**: Countermeasure scores (higher = more likely spoof)

## 📁 Project Structure

```
crosslingual_bias_audiodeepfake/
├── SSL_Anti-spoofing/           # Main detection framework
│   ├── fairseq-*/              # Modified fairseq for wav2vec2
│   ├── core_scripts/           # Data I/O and utilities
│   ├── models/                 
│   │   ├── aasist.py          # AASIST implementation
│   │   └── wav2vec2_ssl.py    # wav2vec2 front-end
│   └── main.py                 # Training script
├── evaluate_mlaad/             # Cross-lingual evaluation
│   ├── main.py                 # Evaluation entry point
│   ├── config.py              # Configuration
│   ├── data.py                # MLAAD data handling
│   ├── evaluate.py            # Bias analysis
│   └── visualize.py           # Generate paper figures
├── scripts/
│   └── statistical_tests.py   # Mann-Whitney U, CLES
├── configs/                    # Configuration files
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Download Data

```bash
# Download ASVspoof 5 Track 01 (registration required)
# https://www.asvspoof.org/

# Download MLAAD dataset
# https://github.com/piotrkawa/mlaad
# Extract only the MMS subset for evaluation
```

### 2. Prepare Data

```bash
python scripts/prepare_data.py \
    --asvspoof_dir /path/to/asvspoof5 \
    --mlaad_dir /path/to/mlaad \
    --output_dir data/
```

### 3. Train Model (Optional - pretrained available)

```bash
python SSL_Anti-spoofing/main.py \
    --config configs/aasist_wav2vec2.yaml \
    --train_data data/asvspoof5/train \
    --val_data data/asvspoof5/dev \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4
```

### 4. Evaluate Cross-Lingual Bias

```bash
cd evaluate_mlaad/
python main.py \
    --model_path ../models/aasist_wav2vec2_asvspoof5.pth \
    --mlaad_path ../data/mlaad_mms/ \
    --languages fi,de,ru,sw,uk,en,fr,nl,hu,ro \
    --output_dir ../results/
```

## 📈 Results

### Detection Performance by Language

| Language | Code | Mean CM Score | Std Dev | Detection Quality |
|----------|------|--------------|---------|-------------------|
| Romanian | ro | 0.99 | 0.05 | ✅ Excellent |
| French | fr | 0.97 | 0.15 | ✅ Excellent |
| Russian | ru | 0.97 | 0.14 | ✅ Excellent |
| Finnish | fi | 0.95 | 0.18 | ✅ Very Good |
| English | en | 0.84 | 0.31 | ⚠️ Moderate |
| German | de | 0.82 | 0.32 | ⚠️ Moderate |
| Dutch | nl | 0.82 | 0.30 | ⚠️ Moderate |
| Hungarian | hu | 0.74 | 0.38 | ⚠️ Variable |
| Swahili | sw | 0.48 | 0.41 | ❌ Poor |
| Ukrainian | uk | 0.12 | 0.27 | ❌ Very Poor |

### Statistical Analysis and plotting resources

```bash
# Generate paper figures
python ./scripts/statistical_tests.py \
    --csv_path <path to results file> \
    --output_dir <desired output folder>
```

The input CSV file is the one with the following columns:

original_file|language|is_original_language|duration|training_data|model_name|architecture|transcript|language_folder|model_folder|meta_csv_path|absolute_audio_path|score|gender

The script writes the following plots to disk:
- Figure 2: Score distributions (violin plots)
- Figure 3: P-value and CLES heatmap

## 🔬 Reproducing Paper Results

### Complete Pipeline

```bash
# 1. Train on ASVspoof 5 (English only)
./scripts/train_english_only.sh

# 2. Evaluate on MLAAD MMS subset
./scripts/evaluate_crosslingual.sh

# 3. Statistical analysis
./scripts/statistical_tests.py
```

### Expected Outcomes
- Model achieves ~5.16% EER on ASVspoof 5 evaluation set
- Significant language-dependent performance gaps
- P-values < 0.001 for most language pairs
- CLES values showing large effect sizes

## 📊 Configuration

Key parameters in `configs/aasist_wav2vec2.yaml`:

```yaml
model:
  frontend: "wav2vec2-xls-r-300m"
  backend: "aasist"
  pooling: "2d_self_attentive"
  
training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-4
  optimizer: "adam"
  loss: "binary_cross_entropy"

evaluation:
  languages: ["fi", "de", "ru", "sw", "uk", "en", "fr", "nl", "hu", "ro"]
  samples_per_language: 1000
  
```

## 📝 Citation

Soon

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## ⚠️ Important Notes

1. **Ethical Considerations**: This research reveals biases that should be addressed in deployment, not exploited
2. **Reproducibility**: Random seeds fixed for reproducibility
3. **Computational Requirements**: ~2-3 hours on single GPU for full evaluation
4. **Data Access**: ASVspoof 5 requires registration; MLAAD is publicly available

## 🙏 Acknowledgments

- CAPES – Finance Code 001
- FAPESP Horus project (Grant #2023/12865-8)  
- FAPESP BI0S project (Grant #2020/09838-0)
- CPQD Company
- Authors of MLAAD dataset (Müller et al., 2024)
- Authors of Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation (Tak et al., 2022)
- ASVspoof challenge organizers

## 📧 Contact

For questions: paulad@unicamp.br

---

**Note**: This work analyzes bias in existing detection systems. The MLAAD dataset was created by Müller et al. (2024) and is used here for controlled evaluation.