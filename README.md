# QLoRA Fine-Tuning of Qwen2-0.5B on Alpaca

Instruction-tunes [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B) on the [Alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset using QLoRA: the base model is loaded in 4-bit, and only small LoRA adapters are trained. It then studies how training-set size and LoRA rank affect perplexity, trainable parameters, and GPU memory.

## What's in the notebook

1. **Baseline**: generates a response from the base model for a test instruction ("Write a Python function that checks whether a number is prime").
2. **Data prep**: shuffles Alpaca-cleaned (seed 42), takes 7,000 examples, and formats each into an `### Instruction / ### Input / ### Response` prompt. 5,000 are used for training and 2,000 for testing.
3. **QLoRA setup**:
   - 4-bit NF4 quantization with double quantization and fp16 compute (`bitsandbytes`).
   - LoRA (r=16, alpha=16, dropout 0.05) on all attention and MLP projections plus `lm_head`.
4. **Training**: `trl`'s `SFTTrainer`, 1 epoch, batch size 8, learning rate 4e-4, paged 8-bit AdamW, max length 512 with sequence packing.
5. **Evaluation**: compares generations before and after fine-tuning, and reports train and test loss and perplexity.
6. **Experiments**:
   - **Training-set size**: 500, 1,000, 2,000 and 5,000 examples (rank 16), measuring perplexity vs. size.
   - **LoRA rank**: r = 4, 8 and 16 (5,000 examples), measuring perplexity, trainable parameters and peak GPU memory.

## Setup

Needs a CUDA GPU (`bitsandbytes` 4-bit loading does not run on CPU or Apple Silicon). A free Google Colab T4 works.

```bash
pip install bitsandbytes transformers peft accelerate datasets trl
```

## Run

Open `QLoRA.ipynb` (e.g. in Colab with a GPU runtime) and run the cells in order. The trained adapter is saved to `./final_lora_adapter`.

## Results

<!-- Add the perplexity-vs-size and perplexity-vs-rank plots, the rank table
     (perplexity, trainable params, peak memory), and a before/after generation here. -->

## References

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
