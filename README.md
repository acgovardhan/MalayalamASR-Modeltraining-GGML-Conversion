# 🎙️ Offline Malayalam ASR — Model Training & GGML Conversion

> Fine-tuning OpenAI Whisper Tiny on Malayalam speech data and converting it to a compressed GGML binary for fully offline, on-device inference on Android.

**Part of the Offline Malayalam ASR project** — a research internship at the [Centre for Development of Imaging Technology (C-DIT)](https://cdit.org), Thiruvananthapuram, Kerala, aimed at building voice technology for low-literacy farming communities in rural Kerala.

📱 **Android App Repository:** [acgovardhan/local-malayalam-asr](https://github.com/acgovardhan/local-malayalam-asr)

---

## 📌 What This Repository Contains

| File | Description |
|------|-------------|
| `finetune-malayalam-asrmodel.ipynb` | Fine-tunes OpenAI Whisper Tiny on the FLEURS Malayalam dataset |
| `model-to-ggml-compression.ipynb` | Converts the trained model to GGML binary format for Android deployment |

Both notebooks are designed to run on **Google Colab** with a free T4 GPU. No local GPU or paid resources are required.

---

## 🧠 What Is This Project?

This project builds an **offline Malayalam speech-to-text system** that runs entirely on an Android smartphone without any internet connection.

The core idea is simple: farmers and elderly citizens in rural Kerala often cannot read or type, but they can speak. Existing voice tools require internet, cost money per call, and have poor Malayalam support. This project addresses all three problems by training a small, efficient model and shipping it directly inside a mobile app.

The pipeline has two stages, one notebook each:

**Stage 1 — Fine-Tuning**
Takes the pre-trained `openai/whisper-tiny` model (which already knows some Malayalam from its original training) and fine-tunes it specifically on the FLEURS Malayalam dataset. This teaches the model to better recognize Malayalam speech patterns and produces a HuggingFace-format model saved to Google Drive.

**Stage 2 — GGML Conversion**
Takes the fine-tuned HuggingFace model and converts it to the GGML binary format using the [whisper.cpp](https://github.com/ggerganov/whisper.cpp) conversion script. GGML is a compressed, CPU-optimized format that reduces the model from ~150 MB to ~75 MB and allows it to run inference on a smartphone without a GPU or internet connection.

The resulting `ggml-model.bin` file is then placed inside the Android app.

---

## 🗂️ Model Details

| Property | Value |
|----------|-------|
| Base model | `openai/whisper-tiny` |
| Parameters | 39 million |
| Architecture | Transformer encoder-decoder |
| Dataset | [google/fleurs](https://huggingface.co/datasets/google/fleurs) — `ml_in` (Malayalam India) |
| Training samples | ~2,700 |
| Validation samples | ~400 |
| Audio format | 16kHz, mono, PCM |
| Training steps | 5,000 |
| Effective batch size | 8 (1 per device × 8 gradient accumulation steps) |
| Learning rate | 1e-5 with 500 warmup steps |
| Mixed precision | FP16 |
| Evaluation metric | Word Error Rate (WER) — lower is better |
| Training platform | Google Colab (T4 GPU, free tier) |
| Estimated training time | 9–16 hours |

---

## ⚙️ How to Reproduce — Step by Step

### Prerequisites

- A Google account
- Google Drive with at least **2 GB free space** (for saving model checkpoints)
- The two notebook files from this repository downloaded to your computer

---

### Step 1 — Open Google Colab

Go to [colab.research.google.com](https://colab.research.google.com)

Upload `finetune-malayalam-asrmodel.ipynb`:
```
File → Upload notebook → select finetune-malayalam-asrmodel.ipynb
```

---

### Step 2 — Set the Runtime to T4 GPU

This is important — without a GPU the training will be extremely slow.

```
Runtime → Change runtime type → Hardware accelerator → GPU → GPU type → T4
```

Click Save.

---

### Step 3 — Run the Training Notebook

Run each cell from top to bottom in order. Here is what each cell does:

| Cell | What it does |
|------|-------------|
| Cell 1 | Installs all required Python libraries |
| Cell 2 | Mounts your Google Drive and creates the save directory |
| Cell 3 | Downloads the FLEURS Malayalam dataset from HuggingFace |
| Cell 4 | Loads Whisper Tiny and configures it for Malayalam fine-tuning |
| Cell 5 | Converts audio to log-Mel spectrograms and tokenizes transcriptions |
| Cell 6 | Sets up the custom data collator for dynamic batching |
| Cell 7 | Defines the WER metric for evaluation |
| Cell 8 | Configures all training hyperparameters |
| Cell 9 | Runs the fine-tuning (this is the long step — 9 to 16 hours) |
| Cell 10 | Saves the final model to Google Drive |

After Cell 10 completes, your trained model will be at:
```
MyDrive/malayalam_whisper_tiny/final/
```

---

### Step 4 — If Your Colab Session Disconnects

Colab free tier has a ~12 hour session limit. If it disconnects during training:

1. Start a new Colab session with T4 GPU
2. Re-run **Cell 1** (install libraries) and **Cell 2** (mount Drive)
3. Re-run **Cells 3, 4, 5, 6, 7, 8** to reload everything
4. In **Cell 9**, change the last line from:
   ```python
   trainer.train()
   ```
   to:
   ```python
   trainer.train(resume_from_checkpoint=True)
   ```
5. Run Cell 9 — it will pick up from the last saved checkpoint automatically

Checkpoints are saved to Drive every 1,000 steps so you will not lose more than 1,000 steps of progress.

---

### Step 5 — Run the Conversion Notebook

Open a **new Colab session** (you do not need a GPU for this step, but it works fine with one).

Upload `model-to-ggml-compression.ipynb`:
```
File → Upload notebook → select model-to-ggml-compression.ipynb
```

Run each cell in order:

| Cell | What it does |
|------|-------------|
| Cell 1 | Clones whisper.cpp and openai/whisper from GitHub |
| Cell 2 | Mounts your Google Drive and sets model/output paths |
| Cell 3 | Copies missing tokenizer files into the model directory |
| Cell 4 | Runs the GGML conversion script |
| Cell 5 | (Optional) Compiles whisper.cpp and validates the model with a test audio file |
| Cell 6 | Shows you where your model is saved |

After Cell 4 completes, your converted model will be at:
```
MyDrive/malayalam_whisper_tiny/ggml/ggml-model.bin
```

Download this file from Google Drive to your computer.

---

### Step 6 — Deploy to Android

Copy `ggml-model.bin` to the Android app repository:

```
local-malayalam-asr/
└── app/
    └── src/
        └── main/
            └── assets/
                └── ggml-model.bin   ← place it here
```

Then open the project in Android Studio, build, and run.

Full Android app instructions: [acgovardhan/local-malayalam-asr](https://github.com/acgovardhan/local-malayalam-asr)

---

## 📊 Expected Results

| Metric | Expected Value |
|--------|----------------|
| Training steps completed | 5,000 |
| Best model checkpoint | Around step 4,000–5,000 |
| WER on validation set | 60–90% |
| GGML model size | ~75 MB |
| RAM required on device | ~200–300 MB |

### Why is WER high?

The Word Error Rate is high and this is **expected and documented**. It is not a bug or a code problem. The root cause is dataset size — FLEURS Malayalam contains only ~2,700 training samples, representing roughly 8–10 hours of audio. Commercial ASR systems are trained on thousands of hours. Malayalam is also morphologically complex with long agglutinative words, which makes WER naturally higher than simpler languages.

The architecture and pipeline are correct. Accuracy will improve significantly with more training data.

---

## 🔭 Limitations & Future Work

- **Dataset size:** FLEURS Malayalam is too small for robust ASR. Adding AI4Bharat IndicSpeech or a community-collected agricultural speech corpus would dramatically improve WER.
- **Model size:** Upgrading to Whisper Base (74M parameters) would improve accuracy while remaining deployable on mid-range phones.
- **Quantization:** Converting to 4-bit GGML (Q4_K_M) would reduce the model to ~35 MB with minimal accuracy loss.
- **Domain adaptation:** A secondary fine-tuning pass on agricultural vocabulary (crop names, pest names, government scheme names) would make the model far more useful for the target users.
- **Dialectal variation:** FLEURS contains standard read Malayalam. Regional dialects (Malabar, Palakkad, Kochi) and conversational speech are not well represented.

---

## 🌾 Why This Matters

Kerala has ~3 million registered farmers. A significant proportion are elderly, have low digital literacy, and live in areas with poor internet connectivity. Every existing voice tool assumes internet access, English proficiency, or the ability to read and type.

This project is a step toward voice interfaces that work for people who have been left out of the digital revolution — not because they lack intelligence or capability, but because the tools were never built for them.

---

## 📚 References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [FLEURS Dataset](https://huggingface.co/datasets/google/fleurs)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [AI4Bharat](https://ai4bharat.iitm.ac.in)

---

## 📄 License

MIT License — free to use, modify, and build upon.

---

*Built during a Research Internship at the Centre for Development of Imaging Technology (C-DIT), Thiruvananthapuram, Kerala — Sep 2025 to Mar 2026.*
