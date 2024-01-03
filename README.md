# Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data
Source code for the EMNLP 2023 Findings paper entitled "Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data" by KaShun SHUM et al.

## Repo structure
### Dataset Format
```
    [
    {
        "id": "1",
        "Question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Rationale": " She sold 48 clips in April. In May she sold half as many, so she sold 48 / 2 = 24 clips. In total she sold 48 + 24 = 72 clips",
        "Answer": "The answer is 72.",
        "Ground_truth": "72"
    },
    ...
    ]
```
### Scripts
The script for running the example selection.

### Src
The implementation for running the example selection.



## Citation

If you use or extend our work, please cite the following [paper](https://aclanthology.org/2023.findings-emnlp.811/):
```
@inproceedings{shum-etal-2023-automatic,
    title = "Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data",
    author = "Shum, Kashun  and
      Diao, Shizhe  and
      Zhang, Tong",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.811",
    doi = "10.18653/v1/2023.findings-emnlp.811",
    pages = "12113--12139",
}
```