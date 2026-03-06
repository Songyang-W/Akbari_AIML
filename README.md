# Akbari AIML — REBOA R01 Machine Learning

Machine learning pipeline for the **REBOA (Resuscitative Endovascular Balloon Occlusion of the Aorta)** R01 project: identifying high-risk hemodynamic periods during cardiac arrest (CA) resuscitation and guiding precision-timed REBOA deployment to improve neurological outcomes.

---

## Goal (AIML rationale)

- **Identify high-risk periods:** Develop and validate ML algorithms that find **high-risk periods of hemodynamic instability**—especially cerebral hypoperfusion—during CPR and **immediately after return of spontaneous circulation (ROSC)** that are most predictive of poor neurological outcome.
- **Guide REBOA deployment:** Use these algorithms to support **data-driven, precision-timed REBOA** deployment during those vulnerable windows, so that transient aortic occlusion can augment cerebral perfusion when it matters most, without relying on empirical timing alone.
- **Leverage multimodal data:** Use existing multimodal neuro-monitoring from >1000 rodent CA experiments (e.g., BP, HR, SpO₂, ECG, EEG, CBF, brain oxygenation/CMRO₂) to train explainable models (e.g., logistic regression, random forests, XGBoost) and to derive interpretable rules for when and how to intervene.

---

## Challenge (AIML rationale)

- **No data-driven REBOA timing:** Current use of REBOA during CA is largely **empirical**; there are no robust, data-driven strategies to optimize **inflation and deflation timing** relative to physiology.
- **Unclear role of post-ROSC dynamics:** It is not well understood whether post-ROSC **hypoperfusion and hyperperfusion** mainly **reflect** the severity of anoxic injury or can **determine** (and be modified to improve) neural recovery—or both. ML on high-fidelity multimodal data can help separate correlation from modifiable targets.
- **Focus on the critical window:** Most ML work in CA prognostication targets **hours after ROSC**. This project targets the **seconds and minutes during CPR and immediately post-ROSC**—the brief window when a device like REBOA can rapidly intervene without the drawbacks of additional pharmacologic vasopressors.

---

## Repository structure

```
Akbari_AIML/
├── README.md
├── data/                 # Input data (see data/README.md)
├── outputs/               # Preprocessing artifacts, figures, models
├── R01_supervised_random_forest.py   # Main pipeline: NDS preprocessing → RF classification
├── R01_nds_exploration.py
├── ml_pipeline.py
└── create_sample_data.py
```

## Setup and run

1. **Environment:** Use a Python environment with `pandas`, `scikit-learn`, `openpyxl`, and optionally `matplotlib` (e.g. `conda activate eightsleep-ml`).
2. **Data:** Place your Excel file in `data/` with sheets: `Filtered_Subset_No_REBOA`, `No_REBOA_selected_column`, `cleaned_no_reboa` (or use CSVs as in `data/README.md`).
3. **Run pipeline:**
   ```bash
   python R01_supervised_random_forest.py
   ```
   Outputs (preprocessing, plots, models) are written to `outputs/`.

See **data/README.md** for exact data formats and options (Excel vs CSV).

---

## Reference

Based on the REBOA R01 Research Strategy (updated draft): *Development and validation of machine learning predictive algorithms that identify high-risk periods of hemodynamic instability and guide ML-based strategies to optimize neurological outcomes* (Specific Aim 3).
