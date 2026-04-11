# Αναγνώριση Οδηγού Από Αισθητήρες

Αυτό το repository ξεκίνησε για την αναπαραγωγή του paper:
`Incorporating Gaze Behavior Using Joint Embedding With Scene Context for Driver Takeover Detection` (ICASSP 2022).

Ο άμεσος στόχος είναι να στηθεί πρώτα ένα καθαρό και λειτουργικό baseline πάνω στο HDBD και μετά να προσαρμοστεί σταδιακά προς τη διπλωματική:
`Αναγνώριση οδηγού από αισθητήρες σε αυτοκίνητο`.

## Πού Βρισκόμαστε

Μέχρι στιγμής το project περιλαμβάνει:

- έλεγχο του dataset HDBD χωρίς πλήρες extraction
- δημιουργία paper-style windows 3 δευτερολέπτων
- baseline labels για takeover detection
- multimodal dataset loader
- baseline model με visual branch, signal branch και HMI features
- training / validation / test pipeline
- metrics όπως `ROC AUC`
- experiment logging και checkpoint saving

Με λίγα λόγια, έχουμε ήδη λειτουργικό baseline pipeline του paper και το επόμενο βήμα είναι τα πιο σοβαρά experiments πριν τη μετάβαση προς τη διπλωματική.

## Δομή Φακέλων

- `configs/`: JSON presets για reproducible experiments
- `data/raw/`: raw αρχεία και cached archives
- `data/interim/`: ενδιάμεσα artifacts όπως window index και signal stats
- `data/processed/`: τελικά processed δεδομένα
- `docs/`: σημειώσεις για το paper και την αναπαραγωγή του
- `experiments/`: τοπικά logs, summaries και checkpoints από runs
- `notebooks/`: πρόχειρα notebooks
- `scripts/`: scripts για inspection, index building, training και checks
- `src/`: ο reusable κώδικας του project
- `tests/`: χώρος για tests
- `outputs/`: τοπικά outputs όπως plots και reports

## Βασικό Workflow

1. Έλεγχος του dataset:

```powershell
cd C:\Users\User\Documents\ΔΙΠΛΩΜΑΤΙΚΗ\driver-sensor-recognition
.\.venv\Scripts\python.exe scripts\inspect_hdbd.py --bundle ..\hdbd.tar.gz
```

2. Δημιουργία window index:

```powershell
.\.venv\Scripts\python.exe scripts\build_paper_window_index.py --bundle ..\hdbd.tar.gz
```

3. Έλεγχος dataset loader:

```powershell
.\.venv\Scripts\python.exe scripts\check_paper_dataset.py --bundle ..\hdbd.tar.gz
```

4. Ανάλυση candidate labels:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_label_candidates.py --bundle ..\hdbd.tar.gz
```

5. Report-only run για split statistics:

```powershell
.\.venv\Scripts\python.exe scripts\train_paper_baseline.py --bundle ..\hdbd.tar.gz --report-only --evaluate-test --limit-train-samples 32 --limit-val-samples 16 --limit-test-samples 16 --subset-strategy balanced --num-split-groups 5 --run-name paper-report
```

6. Μικρό pilot training experiment:

```powershell
.\.venv\Scripts\python.exe scripts\train_paper_baseline.py --bundle ..\hdbd.tar.gz --batch-size 1 --epochs 1 --max-train-batches 4 --max-val-batches 4 --limit-train-samples 8 --limit-val-samples 8 --evaluate-test --max-test-batches 4 --limit-test-samples 8 --subset-strategy balanced --run-name paper-pilot-exp1 --checkpoint-metric val_loss
```

## Experiment Presets

Υπάρχουν πλέον έτοιμα JSON configs στο `configs/` ώστε τα σημαντικά runs να είναι reproducible και να μην ξαναγράφονται κάθε φορά από την αρχή.

Report-only 5-split preset:

```powershell
.\.venv\Scripts\python.exe scripts\train_paper_baseline.py --config configs\paper_report_5split.json
```

Medium readiness run:

```powershell
.\.venv\Scripts\python.exe scripts\train_paper_baseline.py --config configs\paper_medium_readiness.json
```

Full paper-style reproduction run:

```powershell
.\.venv\Scripts\python.exe scripts\train_paper_baseline.py --config configs\paper_full_reproduction.json
```

Μπορείς πάντα να κάνεις override ένα preset από το command line. Παράδειγμα:

```powershell
.\.venv\Scripts\python.exe scripts\train_paper_baseline.py --config configs\paper_medium_readiness.json --num-workers 2 --run-name medium-readiness-workers2
```

Σημείωση:

- Τα `balanced` limited subsets είναι μόνο για debugging και όχι για paper-comparable metrics.
- Το test evaluation πλέον μπορεί να γίνει πάνω στο `best` validation checkpoint με `--test-checkpoint best`.

## Performance Notes

- Τα compressed `tar.gz` archives είναι πρακτικά το μεγαλύτερο I/O bottleneck του baseline.
- Στο πρώτο access δημιουργούνται persistent basename indexes δίπλα στα cached inner archives ώστε τα επόμενα runs να αποφεύγουν επαναλαμβανόμενο archive scanning.
- Το `paper_medium_readiness.json` ενεργοποιεί ήδη `prefetch_subset_assets=true`, γιατί τα medium runs είναι πολύ πιο σταθερά όταν τα πραγματικά αρχεία του subset υλοποιούνται τοπικά πριν το training.
- Για μεγάλα runs, αν έχεις αρκετό local disk, μπορείς να ζητήσεις πλήρες prefetch των active splits:

```powershell
.\.venv\Scripts\python.exe scripts\train_paper_baseline.py --config configs\paper_full_reproduction.json --prefetch-subset-assets --prefetch-active-splits
```

- Το `prefetch-active-splits` μπορεί να χρειαστεί πολύ χώρο στο disk, αλλά είναι ο πιο ρεαλιστικός δρόμος όταν θέλεις να αποφύγεις random access πάνω στα tar archives.

## Τοπικό Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Σημείωση για Windows / PowerShell:

- Σε αυτό το project δεν βασιζόμαστε στο `activate`.
- Ο πιο σταθερός τρόπος είναι να τρέχεις πάντα scripts ως:
  `.\.venv\Scripts\python.exe ...`
- Πριν από κάθε command, βεβαιώσου ότι βρίσκεσαι μέσα στο:
  `C:\Users\User\Documents\ΔΙΠΛΩΜΑΤΙΚΗ\driver-sensor-recognition`

## Χρήσιμα Αρχεία

- `docs/paper_baseline_spec.md`: τι θεωρούμε baseline υλοποίηση του paper
- `docs/paper_reproduction_gap_analysis.md`: τι έχουμε καλύψει και τι απομένει
- `data/interim/paper_window_index_summary.json`: σύνοψη των generated windows
- `data/interim/paper_signal_stats.json`: normalization stats για τα paper signals

## Στόχος Μετά Το Baseline

Αφού σταθεροποιηθεί η αναπαραγωγή του paper, το pipeline θα προσαρμοστεί προς τη διπλωματική ώστε να μεταβούμε από το task του `takeover detection` στο task της `αναγνώρισης οδηγού από αισθητήρες`, με έμφαση σε privacy-preserving σήματα και οδηγικό προφίλ.
