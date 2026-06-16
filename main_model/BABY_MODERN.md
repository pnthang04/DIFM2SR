# Baby Modern BGE/SigLIP Dataset

Keep the archive at:

```text
dataset/baby_modern_bge_siglip.tar.gz
```

Prepare it for RecBole/SGP4SR:

```bash
python scripts/prepare_baby_modern.py
```

This creates:

```text
dataset/baby_modern/
|-- baby_modern.train.inter
|-- baby_modern.valid.inter
|-- baby_modern.test.inter
|-- baby_modern.text
|-- baby_modern.image
`-- metadata/
```

Train on train/valid and evaluate on test:

```bash
python run.py -d baby_modern
```

Or prepare and train in one command:

```bash
python scripts/prepare_baby_modern.py --run-train
```

The archive is about 149 MB. GitHub normal Git pushes reject files over
100 MB, so use Git LFS if this archive must live in GitHub. Plain SSH copy to
a server is fine.
