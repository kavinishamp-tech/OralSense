# Multimodal Artificial Intelligence for Oral Cancer Screening: Integration of Clinical Images, Patient Metadata, Histopathology, Explainability, and Longitudinal Patient Tracking

**Author:**  
**Institution:**  
**Course / Project:** Multimodal Oral Cancer AI  
**Date:** June 2026

---

## Abstract

Oral cancer remains a clinically significant and often late-detected malignancy, particularly in regions where tobacco, alcohol, smokeless tobacco, and areca nut exposure are common. Early identification of suspicious oral mucosal changes is essential because treatment outcomes are substantially better when disease is detected before advanced invasion and regional spread. However, conventional oral screening relies on clinical access, trained personnel, visual examination quality, documentation consistency, and timely referral. Recent advances in deep learning create an opportunity to support clinicians with image-based risk assessment, multimodal decision support, and explainable outputs that can help prioritize patients for further evaluation.

This paper presents a professional research manuscript for an end-to-end oral cancer artificial intelligence system developed as a multimodal clinical decision-support prototype. The system combines four complementary components: clinical oral image classification, metadata-enhanced multimodal prediction, histopathology image classification for oral squamous cell carcinoma screening, and explainability through Grad-CAM and SHAP-style metadata attribution. A Flask-based application integrates trained TensorFlow/Keras models, uncertainty estimation through Monte Carlo dropout, patient scan storage, and longitudinal risk tracking using SQLite. Clinical classes include Normal, Variations from normal, Oral Potentially Malignant Disorders, and Oral Cancer. Histopathology classification is framed as a binary Normal versus OSCC task. The clinical dataset split available in the project contains 1,728 training records, 370 validation records, and 371 test records, with strong class imbalance; for example, only 14 oral cancer records occur in the training CSV, compared with 1,501 Normal records.

The central contribution of the work is not only model training, but the integration of artificial intelligence into a clinically oriented workflow that preserves risk interpretation, uncertainty, explainability, and patient history. The paper describes the system architecture, data preparation, modeling pipeline, application workflow, experimental results, limitations, ethical safeguards, and future research directions. The best generated histopathology CNN evaluated in the repository, `generated_histocnn_004`, achieved a validation AUC of 0.9148 and test AUC of 0.9201 for Normal versus OSCC classification. Because the clinical four-class screening model still requires broader external validation, this manuscript presents the system as a research prototype rather than a clinically deployable diagnostic product.

**Keywords:** oral cancer, OSCC, oral potentially malignant disorders, deep learning, multimodal learning, Grad-CAM, SHAP, histopathology, clinical decision support, Monte Carlo dropout

---

<div style="page-break-after: always;"></div>

## 1. Introduction

Oral cancer is a major public health concern because it is frequently associated with preventable exposures, visible or clinically inspectable lesions, and meaningful differences in outcome between early and late diagnosis. The World Health Organization describes oral cancer as part of the global oral disease burden and emphasizes that many oral conditions are preventable or treatable in early stages. In addition, WHO cancer guidance highlights early detection as a core strategy for improving cancer outcomes. The burden is not evenly distributed across populations. Regions with higher use of tobacco, smokeless tobacco, areca nut, and alcohol experience a disproportionate share of oral cancer morbidity.

The clinical challenge is practical as much as biological. Early oral cancer and oral potentially malignant disorders may present as red, white, ulcerative, nodular, or mixed lesions, and their significance depends on site, duration, patient history, and risk factors. In busy clinical environments, documentation quality and referral consistency can vary. In under-resourced settings, patients may present late because of limited access to specialists, low awareness, or insufficient screening coverage. These gaps motivate the development of decision-support tools that can assist clinicians in triaging suspicious lesions while preserving the role of expert diagnosis and histopathological confirmation.

Artificial intelligence has become increasingly relevant to this problem because oral screening is visually rich. Smartphone and intraoral images can capture mucosal appearance, while histopathology images capture cellular and tissue-level patterns needed for definitive diagnosis. However, image-only models may ignore important clinical context. A lesion in an older patient with tobacco and areca nut exposure has a different risk profile from a similar-appearing lesion in a patient without known risk factors. Multimodal learning therefore offers a natural direction: combine visual evidence with structured patient metadata to produce a more clinically informed risk estimate.

This project implements such a multimodal oral cancer AI system. It includes a clinical image model, a multimodal image-plus-metadata model, a histopathology ensemble, uncertainty estimation, Grad-CAM heatmaps, metadata explanations, and a web interface for prediction and patient scan history. The system is designed as a screening and decision-support prototype, not as an autonomous diagnostic authority. Its intended role is to help clinicians document findings, identify high-risk cases, communicate risk, and track changes over time.

The rest of this paper is organized as follows. Section 2 discusses background and related work. Section 3 defines the research problem and objectives. Section 4 describes the dataset and preprocessing pipeline. Section 5 presents the system architecture. Section 6 explains the model design. Section 7 describes explainability and uncertainty estimation. Section 8 presents the application workflow. Section 9 discusses evaluation and available evidence. Section 10 covers limitations, ethics, deployment considerations, and future work.

---

<div style="page-break-after: always;"></div>

## 2. Background and Related Work

Oral cancer generally includes cancers of the lip, oral cavity, and oropharyngeal region, although definitions vary by epidemiological source. The Global Cancer Observatory reported substantial worldwide incidence and mortality for lip and oral cavity cancer in GLOBOCAN 2022, including 188,438 deaths globally. WHO oral health material identifies oral cancer as among the important global oral health conditions and notes shared modifiable risk factors such as tobacco and alcohol use. CDC guidance similarly highlights tobacco and alcohol as common risk factors for cancers of the oral cavity and pharynx.

A major clinical category related to early detection is oral potentially malignant disorders. OPMDs include lesions and conditions that carry a risk of malignant transformation. They are important because intervention and surveillance before invasive cancer can improve patient outcomes. From an AI perspective, distinguishing OPMD from benign variations and normal mucosa can be difficult because class boundaries are clinically nuanced and image appearance may overlap across categories.

Deep learning has been widely explored in medical imaging tasks, including classification, segmentation, prognosis, and histopathological assessment. A 2024 systematic review of deep learning in oral cancer reported promising performance across classification, object detection, segmentation, and prognostic prediction studies, while also emphasizing methodological limitations such as small datasets, inconsistent validation, and the need for stronger external testing. These concerns are directly relevant to oral cancer AI because dataset size, image acquisition variability, class imbalance, and annotation quality can strongly influence model behavior.

Histopathology is another key domain for AI. Deep learning methods can learn cellular morphology, tissue architecture, keratinization patterns, and other diagnostic cues from digitized microscopic images. However, histopathology models require careful validation because slide preparation, staining, scanning, magnification, and tissue sampling can vary across institutions. For oral squamous cell carcinoma, AI-based histopathology classification may be useful as an adjunctive screening or prioritization tool, but pathologist review remains the diagnostic standard.

Explainability has become central to medical AI because clinicians need more than a class label. Grad-CAM is commonly used to visualize image regions contributing to a convolutional neural network prediction. SHAP-style methods estimate feature contributions for structured inputs. These methods do not prove that a model is clinically correct, but they can help identify whether the model appears to attend to plausible regions or metadata variables. In medical settings, explainability is best treated as a safety and auditing aid rather than as a guarantee of trustworthiness.

The SMART-OM dataset is especially relevant because it provides smartphone-based oral mucosa images with expert annotation. A 2026 Scientific Data article describes SMART-OM as an annotated intraoral image dataset developed to support automated oral disease diagnosis, including normal variations, OPMDs, and oral cancer. The project repository includes SMART-OM-like directory structures and metadata files, making it an appropriate foundation for a multimodal prototype.

---

<div style="page-break-after: always;"></div>

## 3. Research Problem and Objectives

The research problem addressed in this project is the development of a practical AI-assisted oral cancer screening workflow that can combine visual evidence, patient risk factors, histopathology image support, uncertainty estimation, explainability, and longitudinal record keeping. Many AI studies stop at model training. In contrast, this project attempts to connect model predictions to a usable clinical application interface.

The primary research question is:

**How can clinical oral images, patient metadata, and histopathology image models be integrated into a single explainable decision-support system for oral cancer screening and risk tracking?**

The project is guided by five objectives.

First, the system should classify clinical oral images into clinically meaningful categories: Normal, Variations from normal, OPMD, and Oral Cancer. These categories represent a practical screening hierarchy rather than a definitive pathology taxonomy.

Second, the system should incorporate structured patient metadata. The metadata branch includes age, sex, smoking, chewing, areca nut use, and alcohol use. These features are selected because they are clinically relevant, easy to collect, and available in the project pipeline.

Third, the system should include a histopathology prediction pathway. Histopathology images are used for binary Normal versus OSCC-style classification through an ensemble of custom convolutional models.

Fourth, the system should provide uncertainty and explainability. Monte Carlo dropout is used to approximate prediction variability, Grad-CAM is used for clinical image heatmaps, and SHAP-style analysis is exposed for metadata contribution.

Fifth, the system should support clinical workflow continuity. The Flask application stores patient demographics, scan records, risk status, Grad-CAM outputs, and longitudinal changes in a SQLite database.

The intended output is not a replacement for clinical judgment. The system is designed as an assistive screening and triage tool. High-risk predictions should prompt further clinical evaluation, specialist referral, or biopsy when appropriate. Low-risk predictions should not override persistent clinical concern.

---

<div style="page-break-after: always;"></div>

## 4. Dataset and Data Preparation

The project uses multiple forms of data: clinical oral images, structured metadata, and histopathology images. The repository includes CSV split files under `data/`, metadata spreadsheets under `Metadata/`, descriptor spreadsheets under `Descriptors/`, trained models under `models/`, and result artifacts under `results/`.

The available clinical split files contain the following records:

| Split | Normal | Variations | OPMD | OC | Total |
|---|---:|---:|---:|---:|---:|
| Training | 1,501 | 125 | 88 | 14 | 1,728 |
| Validation | 322 | 27 | 18 | 3 | 370 |
| Test | 322 | 27 | 19 | 3 | 371 |

These counts show a severe class imbalance. Normal images dominate all splits, while oral cancer examples are extremely sparse. This imbalance is clinically realistic in screening settings, but it creates methodological difficulty. A model trained on such data may learn to favor the majority class, producing deceptively high accuracy while failing to identify rare high-risk cases. Therefore, sensitivity, recall for OC and OPMD, balanced accuracy, macro-F1 score, and confusion matrix interpretation are more important than raw accuracy alone.

The clinical image pipeline resizes images to 224 by 224 pixels and normalizes pixel values to the range 0 to 1. Image files are organized by class folders. Metadata matching is performed using patient or image identifiers such as SMITA IDs extracted from filenames. When metadata is missing, the implementation assigns a default vector representing approximate neutral values. This allows the multimodal model to run on all images, but it also introduces uncertainty because missing metadata may reduce the clinical meaning of the fused representation.

Structured metadata includes age, sex, smoking, chewing, areca nut, and alcohol. Age is normalized in the application using a bounded transformation based on a 10 to 90 year range. Binary exposure variables are represented numerically. These features are clinically relevant but limited. Important variables such as lesion duration, lesion site, pain, ulceration, induration, prior oral cancer history, HPV status, immune status, socioeconomic factors, and clinician examination findings are not yet included.

For histopathology, the repository includes training scripts and trained model files for binary OSCC screening. Histopathology preprocessing differs from clinical image preprocessing. In the Flask application, histopathology images are resized to 96 by 96 pixels and normalized. Earlier transfer-learning scripts use 224 by 224 pixels for ResNet50 and DenseNet121 workflows, while custom deployed models use 96 by 96. This difference should be documented in future model cards so the deployed preprocessing is clearly tied to each model’s training regime.

Data preparation also includes augmentation. Training scripts use random flipping and, in some histopathology workflows, rotation, width and height shifts, zoom, brightness variation, and class weighting. These augmentations are intended to improve robustness, but in medical imaging they must be used carefully. Some transformations are biologically plausible, such as rotations in histopathology patches, while others may be less appropriate depending on acquisition context and lesion anatomy.

---

<div style="page-break-after: always;"></div>

## 5. System Architecture

The project is implemented as an integrated web-based decision-support prototype. The backend is a Flask application, model inference is performed with TensorFlow/Keras, image handling uses OpenCV and NumPy, frontend pages are plain HTML interfaces, and persistent data is stored in SQLite.

At a high level, the architecture contains six layers:

1. **Input layer:** clinical image upload, histopathology image upload, and patient metadata entry.
2. **Preprocessing layer:** image decoding, RGB conversion, resizing, normalization, metadata normalization, and tensor construction.
3. **Model layer:** image-only clinical model, multimodal image-plus-metadata model, and histopathology ensemble.
4. **Interpretability layer:** Grad-CAM image overlays and SHAP-style metadata attribution.
5. **Risk logic layer:** class mapping, risk category assignment, uncertainty labeling, ensemble aggregation, and final recommendation formatting.
6. **Persistence and workflow layer:** patient registration, scan saving, historical risk comparison, doctor login, reminders, and patient history retrieval.

The main Flask application is contained in `app.py`. The clinical model is loaded from `models/best_model.h5`. The multimodal model is loaded from `models/multimodal_model.h5`. Histopathology models are loaded from files such as `histo_custom_1.keras`, `histo_custom_2.keras`, `histo_custom_3.keras`, and `histo_custom_4.keras`, when present. The application exposes routes for prediction, histopathology ensemble inference, metadata explanation, scan saving, patient history, patient listing, login, and logout.

The frontend includes pages for screening, patient history, reminders, habit tracking, and administration. This is important because real-world AI usefulness depends on workflow fit. A technically accurate model may still fail to help clinicians if it does not support patient follow-up, clear reporting, or repeated review.

The system follows a risk-oriented output design. The clinical classes are mapped to risk levels: Normal is Low, Variations is Low-Med, OPMD is Medium, and OC is High. The final multimodal result is selected by comparing the image-only and metadata-informed outputs and choosing the higher-risk result. This conservative strategy favors patient safety because it avoids downgrading a higher-risk image-only prediction solely because metadata suggests a lower class. However, it may increase false positives and should be evaluated against clinical triage priorities.

---

<div style="page-break-after: always;"></div>

## 6. Clinical Image Model

The clinical image model performs four-class classification over Normal, OC, OPMD, and Variations. It is loaded in the application as `best_model.h5`. Inference uses a 224 by 224 RGB image normalized to floating-point values between 0 and 1. The deployed application runs Monte Carlo style repeated predictions by calling the model with training behavior enabled, allowing dropout layers to remain active if present.

The application uses custom threshold logic rather than simply selecting the maximum softmax probability. For the image-only model, oral cancer probability is checked first, followed by OPMD probability and variation probability. If OC probability is at least 0.22, the result is OC. If OPMD probability is at least 0.32, the result is OPMD. If Variations probability is at least 0.32, the result is Variations. Otherwise, the result is Normal.

This thresholding approach reflects a screening mindset. In a screening tool, the highest-probability class may not always be the safest clinical decision if the model produces moderate probability for a serious condition. Lower thresholds for high-risk classes can increase sensitivity, especially for rare oral cancer examples. However, thresholds must be calibrated carefully. They should be selected using validation data, receiver operating characteristic analysis, precision-recall curves, and clinically acceptable tradeoffs between false negatives and false positives.

A key limitation is the extreme scarcity of oral cancer examples in the split CSV files. With only 14 OC records in training and 3 each in validation and test, the four-class clinical model cannot be considered robustly validated for oral cancer detection. In such a setting, reporting a single test accuracy would be misleading. The model should instead be assessed with class-specific recall, specificity, confidence intervals, and external datasets. Additional OC and OPMD cases are necessary to support stronger claims.

The clinical image model is still useful as a prototype because it demonstrates the end-to-end pipeline: image upload, preprocessing, prediction, risk mapping, uncertainty estimation, Grad-CAM visualization, and storage. As a research artifact, it provides a foundation for future data expansion and validation.

---

<div style="page-break-after: always;"></div>

## 7. Multimodal Model

The multimodal model combines image features with structured patient metadata. Its architecture, based on the training script, uses MobileNetV2 as the image feature extractor. The base model is initialized with ImageNet weights and used without its classification head. Earlier layers are frozen while the final layers are fine-tuned. The image branch applies global average pooling, a dense layer with ReLU activation, and dropout.

The metadata branch accepts six features: age, sex, smoking, chewing, areca nut, and alcohol. It passes these through dense layers with ReLU activation and batch normalization. The image and metadata embeddings are then concatenated, passed through another dense layer with dropout, and finally classified using a four-unit softmax layer.

This design is appropriate for a first multimodal prototype because it treats image information and patient risk factors as complementary evidence. Image features capture lesion appearance, while metadata captures prior risk context. The use of MobileNetV2 is also practical because it is computationally lighter than many larger convolutional networks, making it more suitable for local or low-resource deployment.

The deployed multimodal model uses a different class order from the image-only model. The image-only class order is Normal, OC, OPMD, Variations, whereas the multimodal class order is Normal, Variations, OPMD, OC. The application handles this explicitly through separate class-name arrays and separate threshold functions. This is a crucial implementation detail because class-order mismatch is a common and dangerous source of medical AI errors.

The final result returned to the user is the higher-risk class between image-only and multimodal predictions. This logic acknowledges that metadata fusion should not automatically overrule image-based concern. For example, a suspicious lesion image should not be downgraded because the entered metadata lacks tobacco or alcohol exposure. Conversely, metadata may elevate concern when image evidence is ambiguous.

Future versions should evaluate whether this rule improves clinical safety compared with calibrated probabilistic fusion. The present rule is transparent and conservative, but it may not be statistically optimal. A validation study should compare image-only, metadata-only, multimodal, and final risk-rule outputs using sensitivity, specificity, positive predictive value, negative predictive value, calibration, and decision-curve analysis.

---

<div style="page-break-after: always;"></div>

## 8. Histopathology Ensemble

Histopathology provides tissue-level evidence and is central to definitive cancer diagnosis. The project includes a histopathology prediction endpoint that accepts an uploaded image and runs an ensemble of trained custom convolutional models. The deployed models are named CustomCNN_1, CustomCNN_2, CustomCNN_3, and CustomCNN_4 in the application when corresponding files are available.

The histopathology endpoint preprocesses images to 96 by 96 RGB arrays normalized to 0 to 1. For each model, Monte Carlo prediction is run multiple times. The endpoint computes the mean OSCC probability and uncertainty for each model, then averages model outputs to produce an ensemble probability. If the ensemble mean is at least 0.35, the endpoint classifies the image as OC; otherwise, it classifies it as Normal. The output includes class, confidence, OSCC probability, uncertainty, uncertainty label, per-model votes, number of models used, and a recommendation statement.

Ensembling is a reasonable strategy for histopathology because individual neural networks can vary in decision boundaries, especially with limited training data. Averaging can reduce variance and produce more stable predictions. Monte Carlo dropout adds another uncertainty signal, which is helpful when a patch is ambiguous, out of distribution, poorly focused, or different from the training distribution.

However, histopathology AI has strict validation requirements. Patch-level classification does not necessarily equal slide-level or patient-level diagnosis. A biopsy may contain heterogeneous regions, and cancer may be focal. The system should therefore be described as histopathology screening support, not as a replacement for pathologist review. Future versions should specify magnification, stain, scanner or microscope acquisition method, patch sampling strategy, and whether labels are patch-level, slide-level, or patient-level.

For research reporting, histopathology evaluation should include sensitivity and specificity for OSCC, area under the ROC curve, area under the precision-recall curve, confusion matrix, and calibration. If the system is used for triage, sensitivity may be prioritized; if used for workload reduction, specificity and negative predictive value become more important. Any clinical claim should be based on external validation across different staining protocols and acquisition devices.

---

<div style="page-break-after: always;"></div>

## 9. Explainability and Uncertainty

The system includes two explainability mechanisms: Grad-CAM for clinical images and SHAP-style explanations for metadata. Both are valuable because medical AI outputs should be interpretable enough for review, auditing, and communication.

Grad-CAM highlights image regions that contribute strongly to the model’s prediction. In the application, Grad-CAM is generated using the final convolutional layer named `Conv_1` from the image model. The heatmap is normalized, sharpened, resized, blurred, colorized, and overlaid on the original image. The resulting base64-encoded image can be displayed in the frontend and stored with the scan record.

A Grad-CAM overlay can help a clinician judge whether the model is focusing on a plausible lesion region or on irrelevant artifacts such as lighting, retractors, background, lips, teeth, or image borders. However, Grad-CAM has limitations. It is low resolution, may be unstable, and does not prove causality. A visually plausible heatmap does not guarantee a correct prediction, and an implausible heatmap may reveal a shortcut learned by the model. Therefore, Grad-CAM should support review rather than justify unquestioned acceptance.

The metadata explanation endpoint uses SHAP KernelExplainer with a small background set of representative metadata profiles. It predicts from metadata while using a dummy image input, then returns SHAP values for age, sex, smoking, chewing, areca nut, and alcohol. This can help identify which risk factors are pushing a metadata-informed prediction toward a specific class.

The system also uses uncertainty estimation. For clinical and multimodal prediction, the model is run repeatedly with stochastic behavior enabled. Mean probability is used for prediction, and standard deviation is used as an uncertainty proxy. For binary histopathology models, the same approach is applied to OSCC probability. Uncertainty labels are mapped to Very Low, Low, Moderate, and High.

Uncertainty estimation is particularly useful in screening. High-confidence high-risk predictions should be escalated, but high-uncertainty predictions also deserve attention because they may reflect poor image quality, unfamiliar lesion appearance, or model uncertainty near class boundaries. A mature system should present uncertainty alongside image quality checks and allow clinicians to request repeat imaging.

---

<div style="page-break-after: always;"></div>

## 10. Application Workflow

The Flask application translates the AI models into a practical workflow. A clinician can upload an oral image, enter patient metadata, receive image-only and multimodal predictions, inspect a Grad-CAM heatmap, and save the result to a patient record. Separate pages support patient history, reminders, habit tracking, and administration.

The `/predict` endpoint accepts an image and metadata fields. It preprocesses the image twice, once for the image-only model and once for the multimodal model. It normalizes age, constructs the metadata vector, runs Monte Carlo predictions, applies threshold-based class selection, maps classes to risk labels, chooses the higher-risk final output, generates Grad-CAM, and returns a structured JSON response.

The `/predict_histo_ensemble` endpoint accepts a histopathology image and runs all available histopathology models. It returns ensemble probability, uncertainty, and per-model vote information. This provides more transparency than returning only a single class label.

The `/save_scan` endpoint stores patient details and prediction outputs in SQLite. It also compares the current risk category with the previous scan to determine whether risk has increased. This longitudinal feature is clinically meaningful because oral lesions may evolve, regress, or persist over time. Tracking risk trends can support recall scheduling and follow-up.

The `/patient_history/<patient_id>` endpoint retrieves patient demographics and scan history. The system marks whether risk increased or decreased relative to prior visits. This creates a foundation for clinical audit trails and patient monitoring.

Authentication is implemented through a simple doctor login system. The current implementation is suitable for a prototype, but deployment would require stronger password handling, role-based access control, secure session configuration, audit logging, encryption, and compliance with local health-data regulations.

The workflow design reinforces an important principle: medical AI should be embedded in care processes, not isolated as a one-off classifier. Prediction, explanation, reporting, storage, and follow-up need to work together if the system is to provide real clinical value.

---

<div style="page-break-after: always;"></div>

## 11. Evaluation Strategy

The repository contains result artifacts such as training curves, confusion matrix images, Grad-CAM outputs, multimodal training plots, and generated histopathology model manifests. For the generated histopathology CNN experiments, structured AUC and F1 metrics are available in `results/manifests/`. Additional Accuracy, Precision, Recall, ROC, and confusion matrix outputs were generated for the strongest manifest-backed model, `generated_histocnn_004`. For the clinical four-class oral-image workflow, stronger structured reporting is still needed before making performance claims.

The recommended evaluation strategy includes four levels.

**First, technical validation** should confirm that preprocessing, class order, model loading, threshold logic, and endpoint outputs are correct. Unit tests should verify that each model receives tensors of the expected shape and that class mappings remain consistent. This is especially important because the image-only and multimodal models use different class orders.

**Second, internal validation** should use the existing validation and test splits. Because the dataset is imbalanced, reporting should include per-class precision, recall, F1-score, macro-F1, weighted-F1, balanced accuracy, confusion matrices, and one-vs-rest ROC and precision-recall curves. The oral cancer and OPMD classes should be emphasized because they drive clinical risk.

**Third, calibration analysis** should assess whether predicted probabilities correspond to observed frequencies. Calibration curves, expected calibration error, Brier score, and threshold analysis are useful. Medical screening systems should not only rank cases but also communicate risk meaningfully.

**Fourth, external validation** should test the system on images from different devices, clinics, operators, lighting conditions, patient populations, and histopathology preparation protocols. External validation is essential before deployment because medical imaging models often degrade when applied outside their training distribution.

For the current dataset split, class imbalance is the dominant evaluation concern. With only three OC cases in the validation split and three OC cases in the test split, one misclassification changes class-specific recall by 33.3 percentage points. Such small denominators make confidence intervals very wide. Therefore, future work should expand OC and OPMD cases, use patient-level splitting to prevent leakage, and report uncertainty intervals around metrics.

A prospective study would be the most clinically meaningful evaluation. In such a study, clinicians would use the system during routine screening, and outcomes would be compared against expert assessment, biopsy where indicated, and follow-up. The system should be evaluated for diagnostic performance, referral appropriateness, time savings, usability, clinician trust, and patient outcomes.

---

<div style="page-break-after: always;"></div>

## 12. Experimental Results

The strongest generated histopathology CNN model identified in the experiment manifests was `generated_histocnn_004`. This model was trained as a binary Normal versus OSCC classifier using a stratified split of 3,894 training images, 649 validation images, and 649 test images. The model used a probability threshold of 0.50, with OSCC treated as the positive class.

The initial manifest reported only AUC and F1-score. To strengthen the experimental section, additional metrics were computed from the saved model and the same stratified validation and test split. Table 2 summarizes the resulting performance.

| Metric | Validation | Test |
|---|---:|---:|
| AUC | 0.9148 | 0.9201 |
| Accuracy | 0.8228 | 0.8043 |
| Precision | 0.7803 | 0.7625 |
| Recall | 0.9169 | 0.9050 |
| F1-score | 0.8431 | 0.8277 |
| Macro Precision | 0.8348 | 0.8170 |
| Macro Recall | 0.8190 | 0.8003 |
| Macro F1-score | 0.8198 | 0.8006 |

The results show high discrimination, with AUC values above 0.91 on both validation and test sets. Recall for the OSCC class was also high, reaching 0.9169 on validation and 0.9050 on test. This is important for screening because false negatives are clinically more concerning than false positives. Precision was lower than recall, indicating that the model tends to produce some false-positive OSCC predictions. In a screening context this may be acceptable if the system is used to prioritize review rather than make final diagnoses, but the tradeoff should be calibrated with clinical input.

| Split | True Normal predicted Normal | True Normal predicted OSCC | True OSCC predicted Normal | True OSCC predicted OSCC |
|---|---:|---:|---:|---:|
| Validation | 225 | 87 | 28 | 309 |
| Test | 217 | 95 | 32 | 305 |

The validation confusion matrix contains 28 false negatives and 87 false positives. The test confusion matrix contains 32 false negatives and 95 false positives. The high number of true OSCC detections supports the model's sensitivity-oriented behavior, while the false-positive count suggests that threshold tuning or calibration may improve clinical usability.

![ROC Curve for generated_histocnn_004](results/generated_histocnn_004_roc_curve.png)

**Figure 1.** ROC curves for `generated_histocnn_004` on validation and test splits. The validation AUC is 0.9148 and the test AUC is 0.9201.

![Test Confusion Matrix for generated_histocnn_004](results/generated_histocnn_004_confusion_matrix.png)

**Figure 2.** Test confusion matrix for `generated_histocnn_004` using a 0.50 OSCC probability threshold.

These findings significantly strengthen the experimental evidence for the histopathology component. Nevertheless, they should be interpreted as image-level or patch-level results for a generated CNN experiment, not as proof of patient-level diagnostic performance. External validation across different scanners, stains, magnifications, and institutions remains necessary.

---

<div style="page-break-after: always;"></div>

## 13. Results and Interpretation

The implemented system successfully demonstrates an integrated oral cancer AI workflow. It loads trained clinical and multimodal models, processes image and metadata inputs, estimates uncertainty, generates Grad-CAM overlays, runs a histopathology ensemble when model files are present, and stores scan history in a patient database.

The available artifacts indicate that model development included training curves, confusion matrix visualizations, Grad-CAM examples, multimodal training plots, generated model manifests, and extended histopathology evaluation outputs. The generated histopathology CNN results are now reported with AUC, accuracy, precision, recall, F1-score, ROC curves, and confusion matrices. For publication-quality reporting, the same level of metric detail should also be generated for the clinical image-only and multimodal four-class models.

The most important interpretation from the reviewed repository is that the prototype is clinically thoughtful but not yet clinically validated. Its strengths include multimodal fusion, conservative risk selection, uncertainty estimation, explainability, longitudinal tracking, and promising histopathology discrimination. Its limitations include clinical class imbalance, incomplete structured reporting for all deployed models, uncertain external generalizability, simple authentication, and the need for clearer dataset provenance and preprocessing documentation.

The threshold strategy is clinically understandable. By assigning OC when probability exceeds 0.22 and OPMD when probability exceeds 0.32, the system attempts to detect high-risk categories even when the model’s top softmax class might be Normal. This may improve sensitivity for rare but important classes. However, threshold values should be justified empirically. They should be selected through validation curves and reviewed by clinical experts.

The metadata branch adds risk-factor awareness, but metadata must be accurate. Self-reported smoking, alcohol, chewing, and areca nut use can be underreported. Age and sex are simple to collect, but other clinically important variables are absent. The metadata model should therefore support, not dominate, image-based and clinician-based assessment.

The histopathology ensemble adds a second diagnostic context. This is useful because clinical images and histopathology images represent different biological scales. Clinical images show lesion surface appearance; histopathology shows cellular architecture. A mature system could eventually link clinical screening, biopsy recommendation, and pathology support into a unified pathway. At the current stage, the histopathology output should remain advisory.

---

<div style="page-break-after: always;"></div>

## 14. Ethical, Clinical, and Regulatory Considerations

Medical AI systems require careful ethical design. Oral cancer screening tools can influence patient anxiety, referral decisions, clinical workload, and treatment timing. A false negative may delay diagnosis, while a false positive may increase unnecessary referrals or biopsies. Therefore, the system must communicate that it is a screening aid and not a definitive diagnostic tool.

Patient privacy is central. The repository includes a SQLite patient database and stores scan records. In any real deployment, protected health information must be handled according to applicable legal and institutional requirements. This includes access control, encryption at rest and in transit, audit logs, secure backups, consent procedures, and retention policies.

Bias and fairness must also be considered. Oral cancer risk and image appearance may vary across age, sex, geography, ethnicity, socioeconomic status, habits, nutrition, oral hygiene, and imaging device quality. A model trained on one dataset may underperform in another population. Fairness analysis should compare performance across demographic and exposure groups when sample sizes allow.

Explainability can improve clinician review but may also create false reassurance. A heatmap that appears to cover a lesion can be persuasive even if the prediction is wrong. The interface should frame explanations as model-attention aids, not as proof. Clinicians should be trained to use explanations critically.

Regulatory status is another important issue. A system intended for clinical diagnosis or triage may be regulated as software as a medical device, depending on jurisdiction and claims. Before deployment, the development team would need formal risk management, documentation, validation, cybersecurity review, usability testing, and post-market monitoring.

Finally, the system should preserve human authority. The recommended clinical framing is: “AI-assisted risk screening output; clinical examination and specialist review remain necessary.” This language protects patients and clinicians while allowing the technology to support earlier recognition.

---

<div style="page-break-after: always;"></div>

## 15. Limitations

The first limitation is dataset imbalance. The available split files show very few OC examples. This makes robust oral cancer detection difficult and prevents strong claims about model sensitivity. Additional labeled OC and OPMD cases are required.

The second limitation is incomplete structured performance reporting across all model families. The generated histopathology CNN now has machine-readable extended metrics, ROC, and confusion matrix outputs, but the same reporting standard should be applied to the clinical image-only model, multimodal model, and deployed histopathology ensemble.

The third limitation is possible dataset leakage. If multiple images from the same patient appear across training, validation, and test splits, performance estimates may be inflated. Patient-level splitting should be enforced.

The fourth limitation is metadata incompleteness. Missing metadata is replaced with a default vector, which may reduce the validity of multimodal predictions. Missingness should be encoded explicitly, and imputation strategies should be evaluated.

The fifth limitation is external generalizability. Images collected under different lighting, cameras, angles, compression, and clinical protocols may differ substantially from training data. External validation is mandatory.

The sixth limitation is simplified security. The prototype login and database design are appropriate for development but not sufficient for clinical deployment.

The seventh limitation is histopathology context. Patch-level binary prediction cannot replace full slide interpretation by a pathologist. The histopathology model should be validated with clear slide-level protocols.

The eighth limitation is interpretability reliability. Grad-CAM and SHAP explanations are useful but approximate. They should be audited for stability and clinical plausibility.

---

<div style="page-break-after: always;"></div>

## 16. Future Work

Future work should begin with data expansion. The project needs more OC and OPMD cases, preferably collected across multiple institutions, devices, and patient populations. Data collection should include patient-level identifiers for leakage-safe splitting and richer clinical variables such as lesion site, duration, symptoms, prior lesions, visual descriptors, and clinician impression.

The model pipeline should be strengthened with robust evaluation. Automated scripts should generate reproducible reports after every training run. These reports should include confusion matrices, per-class metrics, calibration plots, threshold curves, and confidence intervals. Model cards should document intended use, training data, preprocessing, limitations, and known failure modes.

The multimodal fusion strategy can also be improved. Instead of a simple concatenation network alone, future experiments could compare late fusion, attention-based fusion, calibrated risk models, metadata-only baselines, and uncertainty-aware decision rules. The final risk selection rule should be clinically validated.

Image quality assessment should be added. Blurry, poorly illuminated, overexposed, underexposed, or incorrectly framed images should be flagged before prediction. This would reduce unreliable outputs and encourage repeat capture.

The histopathology pathway should be expanded with slide-level aggregation, stain normalization, magnification metadata, and pathologist review interfaces. Multiple instance learning may be appropriate for whole-slide or multi-patch workflows.

Security and deployment should be improved through environment-based secrets, encrypted storage, secure authentication, Docker packaging, logging, and role-based access control. If the system is deployed in a clinical environment, it should follow institutional IT and medical device governance.

Finally, a prospective clinical usability study should evaluate whether the system improves documentation quality, referral appropriateness, follow-up adherence, and clinician confidence without increasing unsafe reliance on AI.

---

<div style="page-break-after: always;"></div>

## 17. Conclusion

This paper presented a professional research manuscript for a multimodal oral cancer AI screening prototype. The system integrates clinical oral image classification, patient metadata fusion, histopathology ensemble prediction, Grad-CAM visualization, SHAP-style metadata explanation, uncertainty estimation, and longitudinal patient tracking within a Flask application.

The project's strongest contribution is its end-to-end clinical orientation. It does not treat oral cancer AI as only a classification problem; it connects prediction to risk interpretation, explanation, storage, and follow-up. This makes the work more realistic as a decision-support prototype.

At the same time, the system should be presented carefully. The available clinical dataset split is highly imbalanced, especially for oral cancer cases. The generated histopathology CNN results are promising, but the clinical image-only and multimodal pathways still require rigorous structured reporting and external validation. Therefore, the correct scientific framing is that this is a promising prototype requiring broader validation, clinical review, and ethical safeguards before real-world use.

With expanded data, patient-level validation, stronger reporting, calibrated thresholds, and secure deployment, the system could become a meaningful support tool for oral cancer screening workflows. Its future value will depend not only on model accuracy, but on whether it helps clinicians detect risk earlier, document findings better, and follow patients more reliably.

---

## References

1. World Health Organization. **Oral health fact sheet.** WHO. https://www.who.int/news-room/fact-sheets/detail/oral-health

2. World Health Organization. **Cancer fact sheet.** WHO. https://www.who.int/news-room/fact-sheets/detail/cancer

3. International Agency for Research on Cancer. **Global Cancer Observatory: Lip, oral cavity fact sheet, GLOBOCAN 2022.** https://gco.iarc.who.int/media/globocan/factsheets/cancers/1-lip-oral-cavity-fact-sheet.pdf

4. Centers for Disease Control and Prevention. **About Oral Cancer.** CDC Oral Health. https://www.cdc.gov/oral-health/about/about-oral-cancer.html

5. World Health Organization. **Comprehensive assessment of evidence on oral cancer prevention released.** https://www.who.int/news/item/29-11-2023-comprehensive-assessment-of-evidence-on-oral-cancer-prevention-released-29-november-2023

6. Deep learning in oral cancer: a systematic review. **BMC Oral Health**. 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC10859022/

7. Global variations and socioeconomic inequalities in lifetime risk of lip, oral cavity, and pharyngeal cancer: a population-based systematic analysis of GLOBOCAN 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC12165512/

8. Deep learning in cancer genomics and histopathology. **Genome Medicine**. 2024. https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-024-01315-6

9. A Smartphone-based Comprehensive Dataset of Annotated Oral Cavity Images for Enhanced Oral Disease Diagnosis. **Scientific Data**. 2026. https://www.nature.com/articles/s41597-026-06954-5

10. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.** International Journal of Computer Vision. 2020.

11. Lundberg SM, Lee SI. **A Unified Approach to Interpreting Model Predictions.** Advances in Neural Information Processing Systems. 2017.

12. Project repository artifacts reviewed locally: `app.py`, `project_report.md`, `notebooks/train_multimodal.py`, `notebooks/evaluate_multimodal.py`, `train_histopath.py`, `train_generated_models.py`, `data/train.csv`, `data/val.csv`, `data/test.csv`, `models/`, `results/manifests/generated_histocnn_004.json`, and `results/generated_histocnn_004_extended_metrics.json`.
