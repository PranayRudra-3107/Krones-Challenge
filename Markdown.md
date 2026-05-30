1st Krones Vision AI Challenge
A binary image‑analysis challenge to decide whether an image of a returnable bottle shows a bottle fit for reuse or one that must be discard


1st Krones Vision AI Challenge

Submit Prediction
Overview
Welcome to the 1st Krones Vision AI Challenge — a binary image classification competition focused on determining whether a returnable bottle is intact and suitable for reuse. In this context, intact and suitable for reuse means the bottle shows no major cracks or chips, no contaminations or foreign objects, no residual liquid, and no other defect that would prevent it from being safely refilled.

Your task is to develop a model that takes an image of a bottle and outputs a single operational decision:

✅ Reusable or ❌ Not Reusable

You may use classification, object detection, segmentation, anomaly detection, or any other method that ultimately produces a binary decision. The final output must be binary because, in production, each bottle must be either accepted (reused) or rejected (not reused) at the inspection gate—there is no third option.

Top solutions may be evaluated on the Krones Linatronic AI, our industrial Empty Bottle Inspector (EBI) deployed on high‑speed lines. The Linatronic AI uses deep‑learning–based vision to detect cracks, contamination, foreign objects, or remaining liquid, while reliably distinguishing true defects from harmless variations like water droplets or foam. This robustness is typically harder to achieve with classical image processing methods that rely on manual parameter tuning, which can be brittle across different bottles, lighting conditions, and camera setups. Your models aim to improve or complement our existing deep-learning-based approach to further increase inspection accuracy and stability.

Below is a short video showing the Krones Linatronic AI in real operation. It gives you a quick impression of how bottles are inspected on the actual production line. 

Participants will compete for three monetary prizes (see the Prizes section) and the opportunity to contribute to a real‑world industrial inspection system.

This challenge is designed exclusively for students from our partner university, Technische Hochschule Deggendorf (THD), and only currently enrolled THD students are eligible to participate.
Participants must also carefully review and adhere to all competition rules, including data‑usage and privacy requirements.

The provided dataset is strictly for use within this competition. It may not be shared, redistributed, or used for any external purpose.
Competition notebooks, code, and model outputs may not be published or made publicly available outside the Kaggle competition environment unless explicitly permitted by the organizers.
Any violation of these rules may result in disqualification.
Good luck—we’re excited to see your solutions!

Start

a month ago
Close

17 days to go
Evaluation
Final Ranking Criteria
Your final ranking is determined by three components:

1) Model Performance (50%)
Metric: F1-score (binary) on a hidden test set.
2) Model Efficiency (30%)
To fairly evaluate computational efficiency without being affected by extreme outliers or hardware differences, we use a percentile‑based runtime normalization.

✅ How Efficiency Is Measured
All finalists submit:
their trained model,
and an inference script compatible with the official evaluation notebook.
Organizers run every model in an identical Kaggle Notebook environment
(same hardware, batch size, preprocessing, and evaluation script).
We measure the total inference runtime on the provided evaluation set.
✅ Why Percentile-Based?
Raw runtime can contain extreme outliers (e.g., a bug causing very slow inference).
Using the 10th and 90th percentiles creates a stable and fair scoring scale.

✅ Efficiency Score Formula
Let:

t_p10 = 10th percentile runtime (fast models)
t_p90 = 90th percentile runtime (slow models)
t_model = your model’s measured runtime
Then:

If t_model ≤ t_p10 → Efficiency = 1
If t_model ≥ t_p90 → Efficiency = 0
Otherwise:

This ensures:

Fast models score near 1
Slow models score near 0
Outliers do not distort the scale for everyone else
3) Most Valuable Insight (20%)
What’s judged: Quality of your technical insight, analysis, or innovative idea.
Judges: Expert panel from Krones and Technische Hochschule Deggendorf (THD).
Combined Ranking Formula
Your final score is computed using the weighted sum of all three components:


Where:

Performance = F1-score on the hidden evaluation set
Efficiency = Percentile-normalized runtime score
Insight = Panel-assigned score (normalized to 0–1)
All component scores are normalized to a 0–1 range before weighting to ensure fair comparison.

Note:

The hidden test set used for final evaluation is not uploaded to Kaggle and is not identical to the Kaggle test set (which is split into public and private leaderboards).
This ensures a fair and unbiased final ranking.
Important:

The public and private leaderboards on Kaggle display only the Performance (F1) metric and serve solely as an indicator during the competition.
All final metrics (Performance, Efficiency, and Insight) will be evaluated offline by the organizers after the competition closes.
The Final Score (used for prizes and final ranking) will be computed based on these offline evaluations.
Team Formation Rules
✅ Team Formation on iLearn (Mandatory)
Teams had to be formed in advance via the iLearn platform of the THD before joining the challenge.
These iLearn teams are considered the official and final teams for the competition.
✅ Team Formation on Kaggle
For collaboration purposes (e.g. shared notebooks, datasets), teams should also form Kaggle teams.

⚠️ Important:
The Kaggle team must be identical to the team formed on iLearn.

This means:

❌ No new teams
❌ No switching teams
❌ No adding or removing team members
✅ Same members, same grouping as on iLearn
✅ Team Naming Convention
Teams must be named according to their iLearn team number.

Required format:
<iLearn-Team-Number>
Optional custom name may be added in brackets after the number
Examples:

07
12 (EfficientNet Ninjas)
21 (Runtime Optimizers)
The iLearn team number must always be clearly identifiable.

✅ Final Note
Team composition is checked against the iLearn records.
Consistency across platforms is required to ensure fair evaluation and clear attribution.

Submission
Submission Requirements (Important Update ⚠️)
To ensure fair, reproducible, and understandable evaluation, all submissions must include the following components:

✅ What You Must Provide (by the announced deadline)
Each team must share the following via Kaggle with Alexander Hewicker and Max Meierhofer.

⚠️ Important: Exactly one person per team (the team leader) must do the sharing.

Please do not have multiple team members share notebooks.

📓 1. Training Notebook (Required ✅)
You must provide a training notebook that:

Shows how your model was trained
Includes preprocessing, augmentations, and validation strategy
Clearly documents your overall approach
👉 This is important so we can understand how your solution works, not just the final predictions.

🧠 2. Approach Explanation (Required ✅)
Please include a clear explanation of your method, inside the training notebook.

This should cover:

Model architecture
Key design decisions
Use of additional data (e.g., ROI, bottle types, annotations)
Any tricks or optimizations used
📓 3. Evaluation Notebook (Required ✅)
You must submit a Kaggle Notebook that:

Loads your trained model
Runs inference on the test set
Produces predictions in the required submission format
📦 Model Attachment (Critical ✅)
Your trained model (e.g., ONNX file) must be directly attached to your evaluation notebook.

👉 This is critical:

The model file must be available as a notebook dataset/input
The notebook must load the model without any external downloads
We must be able to run your notebook as-is to reproduce predictions
❌ If the model is not attached, we will not be able to evaluate your submission

📌 Example (from Starter Notebook)
Attached ONNX Model

✅ Recommended Approach
Please follow the structure of the provided:

Starter Evaluation Notebook

This notebook demonstrates:

How to attach a model file
How to load it inside the notebook
How to run inference and create a submission
👉 Your submission should follow the same pattern, replacing the sample model with your own.

⚠️ Important Notes
❌ Do not rely on external downloads (e.g., Google Drive, APIs, URLs)
❌ Do not assume internet access during evaluation
✅ Ensure all required files (especially the model) are attached
✅ Your evaluation notebook must run end-to-end without manual intervention
Following these guidelines ensures that all submissions are reproducible, comparable, and fairly evaluated.

✅ Only one shared solution per team will be evaluated.

📅 Final deadline:
15 June 2026, 00:00 (midnight)

If your notebooks are not shared by this time, your solution cannot be evaluated.

✅ Evaluation & Runtime Measurement
Organizers will run the shared evaluation notebook in a fresh, standardized Kaggle environment
Inference runtime will be measured using the timing cell defined in the starter evaluation notebook
F1-score and efficiency are evaluated entirely offline
on a fully hidden test set
not uploaded to Kaggle
If the notebook or model fails to run as shared, scores may be set to 0.

✅ Leaderboard Clarification
Teams may submit as many times as they like to the public Kaggle leaderboard
No sharing with the organizers is required for leaderboard submissions
Public and private leaderboards are for guidance only
Final ranking is based exclusively on offline evaluation
hidden test set
measured inference efficiency
Note on Kaggle verification:
Phone verification is not required by us.
It is enforced solely by Kaggle for notebook sharing.
Organizers never see or access any personal information.

Prizes
🏆 Prize Structure
The competition offers three monetary prizes, awarded to the top‑ranked submissions based on the Final Score (see Evaluation section for details).

Total Prizes Available: 3

Place	Prize
🥇 1st	1500 €
🥈 2nd	1000 €
🥉 3rd	500 €
📌 Additional Details
Prizes are awarded to the team as a whole, not to individual members. Teams are fully responsible for determining and managing their own internal prize distribution.
Winners must meet all eligibility and rule requirements, and may be asked to provide necessary information for prize verification and processing in accordance with applicable laws and organizational policies.



Dataset Description
Data Description
This competition provides image data and annotations for training models that classify empty glass bottles as Reusable (0) or Not Reusable (1).
All images originate from the Krones Linatronic inspection system and reflect real production conditions with natural variation, harmless artifacts, and true defects.

Participants are encouraged to explore the dataset thoroughly — analyzing patterns, variations, defects, and anomalies often leads to deeper insights and stronger solutions.

After reading this page, you should understand:

What files are provided
What each file contains
What format to expect
What you are predicting
What optional metadata is available
Files Provided
1. train_images
Contains 35,342 training images.
Image filenames match the image_id field in train.csv.
Images include a wide variety of bottle types, lighting conditions, and harmless artifacts.
2. test_images
Contains 4418 test images without labels.
Your submission must contain predictions (0 or 1) for all listed image IDs.
3. train.csv
Links each training image to its ground‑truth class:

Column Name	Description
image_id	Image filename (e.g., XXX_000000000001.png)
target	0 = Reusable, 1 = Not Reusable
4. sample_submission.csv
A correctly formatted template for your final predictions.

Column Name	Description
image_id	Test image filename
target	Your predicted class (0 or 1)
5. train_annotations.json (COCO format)
Optional COCO‑style annotations for participants who want to build advanced models (detection, segmentation, multi‑task learning).

Includes:

Segmentation polygons for labeled defects
Bounding boxes
Defect categories
ROI (Region of Interest) per image
Indicates where the bottle appears and can be used for cropping before training
Use these annotations if you want to:

Build detector/segmenter models
Visualize the dataset with tools like FiftyOne
Apply ROI‑based preprocessing
6. test_annotations_roi_only.json (COCO format) (NEW ✅)
We have added an additional file:

Contains ROI (Region of Interest) bounding boxes for all test images
Does NOT include labels, defects, or segmentation annotations
This file was introduced to ensure consistency between training and test pipelines when using ROI-based approaches.

You can use it to:

Apply the same ROI-based cropping strategy used during training
Avoid having to:
Train a separate ROI detector
Use heuristic or fixed-size crops
Build cleaner, more robust inference pipelines
Improve efficiency (smaller inputs, faster inference)
⚠️ Note: This file is purely an addition. No existing files or labels have been modified.

7. bottletypes.csv (NEW ✅)
We have added an additional file:

Maps each image to its corresponding bottle type
Includes both training and test images
Contains:
image_id
bottle_type (string label, e.g., "0,33L Vichy brown crown cap")
split (train or test)
This file was not part of the original dataset but is provided as an optional resource.

You can use it to:

Explore bottle-type distributions across the dataset
Build auxiliary features or multi-input models
Perform more detailed error analysis (e.g., performance by bottle type)
⚠️ Note: Note: This file is purely an addition. No existing files or labels have been modified.

About the ROI (Region of Interest)
Each entry in the COCO annotations includes an ROI (Region of Interest) — this is a bounding region that contains the bottle itself.

You are not required to use the ROI, but it offers several advantages:

Removes irrelevant background
The raw images contain areas that are not part of the bottle (e.g., machine parts, conveyor edges, lighting artifacts).
Cropping to the ROI reduces this background noise and helps the model focus on the bottle only.

Helps model robustness
By reducing distractions, the model is less likely to overfit to background patterns or lighting variations that do not generalize.

Reduces file size & improves speed
Using ROI crops leads to:

Smaller input resolution
Faster training
Faster inference / lower FLOPs (important for the Efficiency Score)
Optional for augmentation workflows
You can train on the full image if your augmentation pipeline benefits from context, but applying the ROI is a strong baseline and often simplifies preprocessing.

With the addition of test_annotations_roi_only.json, you can now apply these benefits consistently to both training and test data.

Feel free to experiment — using or not using the ROI can both be valid approaches depending on your model design.

What You Are Predicting
Your task is to produce a binary classification for each test image:

0 — Reusable (clean, usable bottle)
1 — Not Reusable (contains a defect, contamination, crack, etc.)
Your final submission must follow the structure of the provided sample_submission.csv.

Any method is allowed — classification, detection, segmentation, anomaly detection —
but your final output must be a single binary label per image.

Bottle Types
The dataset includes three common reusable glass bottle types:

0.33L Vichy (brown, crown cap)
0.5L Euro (brown, crown cap)
0.5L NRW (brown, crown cap)
These bottles differ slightly in shape and geometry, which can influence feature appearance (e.g., reflections, wall thickness, curvature). Models should generalize across all types.

Label Categories
The COCO annotations include 26 unique labels (not including the ROI), which map to the two target classes:

0 — Reusable
1 — Not Reusable
The labels fall into three groups:

1. "Good" Labels (Always Reusable)
These conditions are not considered defects.
Images with only these labels belong to the Reusable (0) class.

Embossing
Foam residue
No fault
Water drop
These represent harmless artifacts that the model should learn not to classify as defects.

2. "Conditionally Faulty" Labels
These labels are considered Not Reusable (1) only when the defect is large enough to matter in a real industrial inspection.
Small occurrences are acceptable — large ones must be rejected.

The size threshold (in pixels) for each label is:

Air bubble — faulty only if the area is greater than 500 px
Chip — faulty only if greater than 200 px
Contamination light — faulty only if greater than 180 px
Glass imperfection — faulty only if greater than 100 px
Scuffing — faulty only if greater than 75,000 px
Scuffing heavy — faulty only if greater than 1,200 px
These rules reflect real-world production logic:
tiny imperfections do not affect bottle usability, but larger ones require removal.

3. "Faulty" Labels (Always Not Reusable)
These labels always indicate a defect — any occurrence results in
Not Reusable (1).

Break / Crack
Circlip
Contamination dark
Crown cap
Foil / Semitransparent
Foreign object – manual cleaning
Foreign object – washing machine
Glass shard
Insect
Label
Liquid
Mold
No base visible
Paint residue
Straw
Yeast residue
These represent clear and unambiguous cases where the bottle must be removed from the filling process.

Important Notes About the Dataset
1. Class Imbalance
The dataset is not perfectly balanced:

The number of Reusable (0) vs. Not Reusable (1) samples varies.
Expect skewed distributions — consider techniques like weighted loss, undersampling, oversampling, or robust metrics.
2. Bottle-Type Frequency Imbalance
Some bottle shapes, colors, and manufacturers appear much more frequently than others.
This reflects real production data — models should be robust across rare and common bottle types.

3. Annotation Imperfections
Although great care was taken when creating the COCO annotations:

Occasional labeling mistakes may exist, as is typical with large-scale human annotation.
Participants are encouraged to inspect samples visually and not rely blindly on annotation completeness.
Curiosity and dataset exploration (e.g., using FiftyOne or custom visualization tools) can provide a significant advantage.

Files
39765 files

Size
21.94 GB

Type
png, csv, json

License
Subject to Competition Rules

bottletypes.csv(3.52 MB)

3 of 3 columns


image_id

bottle_type

split
39760

unique values
0,5L NRW brown crown cap
70%
0,33L Vichy brown crown cap
16%
Other (5327)
13%
train
89%
test
11%
ad0f5a12-93fe-4ebb-9aae-ef12ef5246f8_000000000001.png
0,33L Vichy brown crown cap
train
1d1ee6a4-5b1b-4280-a254-29b4cb5c1267_000000000002.png
0,5L Euro brown crown cap
train
3710d5e0-832e-461b-8922-4b83308ae692_000000000003.png
0,33L Vichy brown crown cap
train
9a4a0c6c-4e26-4aba-8d59-f6dfe609fb21_000000000004.png
0,5L NRW brown crown cap
train
799ec8dc-68a5-4690-90ba-7708d016e681_000000000005.png
0,5L NRW brown crown cap
train
0a70ac55-b733-40ff-bd4d-9a27b5b8f17c_000000000006.png
0,5L Euro brown crown cap
train
3f6fd96c-84db-4bc6-9b4a-0f7034bfa338_000000000007.png
0,5L Euro brown crown cap
train
3816e24c-fc66-44aa-91ba-954a1a37d082_000000000008.png
0,5L NRW brown crown cap
train
9f45d60d-85e0-42a4-9c49-a1eb80c8a5a9_000000000009.png
0,5L NRW brown crown cap
train
66d23b5e-cf58-438e-97ce-5cb5624b2fc9_000000000010.png
0,33L Vichy brown crown cap
train
51b6222a-f9cc-414b-ae33-4f214a82353d_000000000011.png
0,5L NRW brown crown cap
train
ca9ec6c8-8f94-44ba-a49a-d7b065b3fe38_000000000012.png
0,5L NRW brown crown cap
train
a0b0047c-04a0-48cd-af41-f4b50756c4d9_000000000013.png
0,5L Euro brown crown cap
train
07eb9c49-bf3e-486a-92b3-9ebeef50f006_000000000014.png
0,5L NRW brown crown cap
train
2c339488-949b-46d8-844e-bd6191f3fab6_000000000015.png
0,5L NRW brown crown cap
train
bc18de21-085f-4a29-bb9d-49addc2c91a3_000000000016.png
0,5L NRW brown crown cap
train
da6cb135-a953-4deb-ad45-de35255cd30a_000000000017.png
0,5L NRW brown crown cap
train
47ae5db0-bb4d-4e42-b4cd-73777c4018ab_000000000018.png
0,5L NRW brown crown cap
train
f093f058-bd61-4ac2-816a-10d952453e19_000000000019.png
0,5L NRW brown crown cap
train
a9b55d84-c761-4957-bcc3-956ecab0fa64_000000000020.png
0,5L Euro brown crown cap
train
318c6988-02af-4f55-9a33-47348ce6e2eb_000000000021.png
0,33L Vichy brown crown cap
train
b10ce6cd-b547-4ced-bb89-ac3eb1997727_000000000022.png
0,33L Vichy brown crown cap
train
1f0d333e-f140-4aad-825e-550906ea37da_000000000023.png
0,5L NRW brown crown cap
train
ff02d7ff-4c4b-43ff-9ed3-0ac9b3e55e21_000000000024.png
0,5L NRW brown crown cap
train
2cfdb9d5-3f4a-4e07-87a6-d5b447aaf0f8_000000000025.png
0,5L NRW brown crown cap
train
eb3aa035-b3e5-4886-bce0-61ef29a795ae_000000000026.png
0,5L NRW brown crown cap
train
ff07aeab-5f71-4fa5-9fd7-598b831664de_000000000027.png
0,5L Euro brown crown cap
train
b5579833-f092-4dc0-afd3-cf5a20be8583_000000000028.png
0,5L NRW brown crown cap
train
8589492c-0aaa-402f-bae5-ff53da66c404_000000000029.png
0,5L NRW brown crown cap
train
831094ac-7e99-4fa8-9bf2-a380909d3a56_000000000030.png
0,5L Euro brown crown cap
train
11ae0936-94d8-4fb0-974b-a4c9e8bcfc60_000000000031.png
0,5L NRW brown crown cap
train
46bf360e-1d05-4ee3-839d-f96750851b66_000000000032.png
0,5L NRW brown crown cap
train
86467c5d-4f7a-4b3d-8663-765c69a4ac25_000000000033.png
0,5L Euro brown crown cap
train
3eff045a-53ca-4bfc-ab01-e3a212921ab1_000000000034.png
0,5L NRW brown crown cap
train
375cacbc-ffff-44ce-9979-de311649a749_000000000035.png
0,5L NRW brown crown cap
train
1dfdd7ad-bfd7-4d4e-b826-8b6e4d97df21_000000000036.png
0,5L NRW brown crown cap
train
efa442f7-b678-4538-b106-b67db7a0aac4_000000000037.png
0,5L Euro brown crown cap
train
08e92781-59a6-4469-a15b-c285010e9807_000000000038.png
0,5L NRW brown crown cap
train
6919a9c0-6c38-410f-8aba-3463ce0aa345_000000000039.png
0,33L Vichy brown crown cap
train
65d6344b-0db5-4aed-a239-4c34cf120401_000000000040.png
0,5L NRW brown crown cap
train
347c02b2-83f3-4f8d-acbb-719840062d0a_000000000041.png
0,5L NRW brown crown cap
train
c82613b3-38e5-40af-9b22-1283d53d18be_000000000042.png
0,33L Vichy brown crown cap
train
1a068707-0dc4-4ff3-aca4-3bad55b4016a_000000000043.png
0,5L NRW brown crown cap
train
28d2bc08-e547-4ec0-b908-2cb8130b90ac_000000000044.png
0,5L NRW brown crown cap
train
bd04bd03-e4a7-4f8b-bf19-4fef51cac58c_000000000045.png
0,5L NRW brown crown cap
train
53cd8990-e837-4901-84b3-b36214bb748c_000000000046.png
0,5L NRW brown crown cap
train
f60ead92-6418-4f41-a92e-4374fce06eb1_000000000047.png
0,5L Euro brown crown cap
train
84231f1a-401d-4582-94e9-be016455f945_000000000048.png
0,5L NRW brown crown cap
train
4bf26171-9921-4106-84b1-af1c8ef4f24a_000000000049.png
0,33L Vichy brown crown cap
train
404b5555-348a-4a61-a230-dfa821dccd5d_000000000050.png
0,5L NRW brown crown cap
train
Data Explorer
21.94 GB

test_images

train_images

bottletypes.csv

sample_submission.csv

test_annotations_roi_only.json

train.csv

train_annotations.json

Summary
39.8k files

7 columns


Download All
Metadata
License
Subject to Competition Rules


Competition Rules
ENTRY IN THIS COMPETITION CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL COMPETITION RULES.
See Section 3.18 for defined terms

The Competition named below is a skills-based competition to promote and further the field of data science. You must register via the Competition Website to enter. To enter the Competition, you must agree to these Official Competition Rules, which incorporate by reference the provisions and content of the Competition Website and any Specific Competition Rules herein (collectively, the "Rules"). Please read these Rules carefully before entry to ensure you understand and agree. You further agree that Submission in the Competition constitutes agreement to these Rules. You may not submit to the Competition and are not eligible to receive the prizes associated with this Competition unless you agree to these Rules. These Rules form a binding legal agreement between you and the Competition Sponsor with respect to the Competition. Your competition Submissions must conform to the requirements stated on the Competition Website. Your Submissions will be scored based on the evaluation metric described on the Competition Website. Subject to compliance with the Competition Rules, Prizes, if any, will be awarded to Participants with the best scores, based on the merits of the data science models submitted. See below for the complete Competition Rules.

You cannot sign up to Kaggle from multiple accounts and therefore you cannot enter or submit from multiple accounts.

1. COMPETITION-SPECIFIC TERMS
1. COMPETITION TITLE
1st Krones Vison AI Challenge

2. COMPETITION SPONSOR
Krones AG

3. COMPETITION SPONSOR ADDRESS
Böhmerwaldstr. 5, 93073 Neutraubling

4. COMPETITION WEBSITE
https://www.kaggle.com/competitions/1st-krones-vision-ai-challenge/

5. TOTAL PRIZES AVAILABLE: €3000
First Prize: €1500 Second Prize: €1000 Third Prize: €500

6. WINNER LICENSE TYPE
Apache 2.0 License

7. DATA ACCESS AND USE
Participants are granted a limited, revocable, non‑exclusive, non‑transferable license to access and use the dataset solely for the purpose of participating in this Competition.

2. COMPETITION-SPECIFIC RULES
In addition to the provisions of the General Competition Rules below, you understand and agree to these Competition-Specific Rules required by the Competition Sponsor:

1. TEAM LIMITS
a. The maximum Team size is five (5). b. Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline. The maximum allowed is the number of Submissions per day multiplied by the number of days the competition has been running.

2. SUBMISSION LIMITS
a. You may submit a maximum of five (5) Submissions per day. b. You may select up to two (2) Final Submissions for judging.

3. COMPETITION TIMELINE
a. Competition Timeline dates (including Entry Deadline, Final Submission Deadline, Start Date, and Team Merger Deadline, as applicable) are reflected on the competition’s Overview > Timeline page.

4. COMPETITION DATA
a. Data Access and Use.

Participants are granted a limited, revocable, non‑exclusive, non‑transferable license to access and use the dataset solely for the purpose of participating in this Competition.

Participants may not: use the data for any purpose outside of this Competition, redistribute, share, publish, or otherwise make the data available to third parties, commercialize the data or use it to develop commercial products or services, attempt to re‑identify any individuals or derive information beyond what is required for Competition tasks.

All rights in the dataset remain with the Competition Sponsor. The license terminates automatically at the end of the Competition unless explicitly extended by the Sponsor.

The Competition Sponsor reserves the right to disqualify any Participant who uses the Competition Data other than as permitted by the Competition Website and these Rules.

b. Data Security.

You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these Rules from gaining access to the Competition Data. You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition. You agree to notify Kaggle immediately upon learning of any possible unauthorized transmission of or unauthorized access to the Competition Data and agree to work with Kaggle to rectify any unauthorized transmission or access.

Participants acknowledge that the Competition Data and any other information disclosed by the Competition Sponsor that is marked as confidential or that a reasonable person would understand to be confidential (“Confidential Information”) is the proprietary property of the Competition Sponsor. Participants agree to treat all such Confidential Information with strict confidentiality and to use it solely for the purpose of participating in this Competition. Participants may not disclose, publish, transmit, or otherwise make available any Confidential Information to any third party, nor use it for any purpose other than developing and submitting a Competition Submission. These confidentiality obligations apply during the Competition and will continue after the Competition ends, unless the Competition Sponsor provides written permission releasing a Participant from these obligations.

5. WINNER LICENSE
a. Under Section 2.8 (Winners Obligations) of the General Rules below, you hereby grant and will grant the Competition Sponsor the following license(s) with respect to your Submission if you are a Competition winner:

[Non-Exclusive: You hereby grant and will grant to Competition Sponsor and its designees a worldwide, non-exclusive, sub-licensable, transferable, fully paid-up, royalty-free, perpetual, irrevocable right to use, reproduce, distribute, create derivative works of, publicly perform, publicly display, digitally perform, make, have made, sell, offer for sale and import your winning Submission and the source code used to generate the Submission, in any media now known or developed in the future, for any purpose whatsoever, commercial or otherwise, without further approval by or payment to you.]

[*Open Source: You hereby license and will license your winning Submission and the source code used to generate the Submission under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code.

For generally commercially available software that you used to generate your Submission that is not owned by you, but that can be procured by the Competition Sponsor without undue expense, you do not need to grant the license in the preceding Section for that software.

In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding Section for that data and/or model(s).

You may be required by the Sponsor to provide a detailed description of how the winning Submission was generated, to the Competition Sponsor’s specifications, as outlined in Section 2.8, Winner’s Obligations. This may include a detailed description of methodology, where one must be able to reproduce the approach by reading the description, and includes a detailed explanation of the architecture, preprocessing, loss function, training details, hyper-parameters, etc. The description should also include a link to a code repository with complete and detailed instructions so that the results obtained can be reproduced.

6. EXTERNAL DATA AND TOOLS
a. You may use data other than the Competition Data (“External Data”) to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness criteria as outlined in Section 2.6.b below. The ability to use External Data under this Section does not limit your other obligations under these Competition Rules, including but not limited to Section 2.8 (Winners Obligations).

b. The use of external data and models is acceptable unless specifically prohibited by the Host. Because of the potential costs or restrictions (e.g., “geo restrictions”) associated with obtaining rights to use external data or certain software and associated tools, their use must be “reasonably accessible to all” and of “minimal cost”. Also, regardless of the cost challenges as they might affect all Participants during the course of the competition, the costs of potentially procuring a license for software used to generate a Submission, must also be considered. The Host will employ an assessment of whether or not the following criteria can exclude the use of the particular LLM, data set(s), or tool(s):

Are Participants being excluded from a competition because of the "excessive" costs for access to certain LLMs, external data, or tools that might be used by other Participants. The Host will assess the excessive cost concern by applying a “Reasonableness” standard (the “Reasonableness Standard”). The Reasonableness Standard will be determined and applied by the Host in light of things like cost thresholds and accessibility.

By way of example only, a small subscription charge to use additional elements of a large language model such as Gemini Advanced are acceptable if meeting the Reasonableness Standard of Sec. 8.2. Purchasing a license to use a proprietary dataset that exceeds the cost of a prize in the competition would not be considered reasonable.

c. Automated Machine Learning Tools (“AMLT”)

Individual Participants and Teams may use automated machine learning tool(s) (“AMLT”) (e.g., Google toML, H2O Driverless AI, etc.) to create a Submission, provided that the Participant or Team ensures that they have an appropriate license to the AMLT such that they are able to comply with the Competition Rules.
7. ELIGIBILITY
a. Unless otherwise stated in the Competition-Specific Rules above or prohibited by internal policies of the Competition Entities, employees, interns, contractors, officers and directors of Competition Entities may enter and participate in the Competition, but are not eligible to win any Prizes. "Competition Entities" means the Competition Sponsor, Kaggle Inc., and their respective parent companies, subsidiaries and affiliates. If you are such a Participant from a Competition Entity, you are subject to all applicable internal policies of your employer with respect to your participation.

8. WINNER’S OBLIGATIONS
a. As a condition to being awarded a Prize, a Prize winner must fulfill the following obligations:

Deliver to the Competition Sponsor the final model's software code as used to generate the winning Submission and associated documentation. The delivered software code should follow [these documentation guidelines][5], must be capable of generating the winning Submission, and contain a description of resources required to build and/or run the executable code successfully. For avoidance of doubt, delivered software code should include training code, inference code, and a description of the required computational environment.
a. To the extent that the final model’s software code includes generally commercially available software that is not owned by you, but that can be procured by the Competition Sponsor without undue expense, then instead of delivering the code for that software to the Competition Sponsor, you must identify that software, method for procuring it, and any parameters or other information necessary to replicate the winning Submission; Individual Participants and Teams who create a Submission using an AMLT may win a Prize. However, for clarity, the potential winner’s Submission must still meet the requirements of these Rules, including but not limited to Section 2.5 (Winners License), Section 2.8 (Winners Obligations), and Section 3.14 (Warranty, Indemnity, and Release).”

b. Individual Participants and Teams who create a Submission using an AMLT may win a Prize. However, for clarity, the potential winner’s Submission must still meet the requirements of these Rules,

Grant to the Competition Sponsor the license to the winning Submission stated in the Competition Specific Rules above, and represent that you have the unrestricted right to grant that license;

Sign and return all Prize acceptance documents as may be required by Competition Sponsor or Kaggle, including without limitation: (a) eligibility certifications; (b) licenses, releases and other agreements required under the Rules; and (c) U.S. tax forms (such as IRS Form W-9 if U.S. resident, IRS Form W-8BEN if foreign resident, or future equivalents).

9. GOVERNING LAW
a. Unless otherwise provided in the Competition Specific Rules above, all claims arising out of or relating to these Rules will be governed by California law, excluding its conflict of laws rules, and will be litigated exclusively in the Federal or State courts of Santa Clara County, California, USA. The parties consent to personal jurisdiction in those courts. If any provision of these Rules is held to be invalid or unenforceable, all remaining provisions of the Rules will remain in full force and effect.

Kaggle Competition Foundational Rules
(Non-editable)

Competition participants must also agree to Kaggle's Foundational Competition Rules. These rules will supersede the competition-specific rules in the event of any conflict.
The following Kaggle Competition Foundational Rules (“ Foundational Rules ”) apply to every competition regardless of whether the Sponsor creates competition-specific rules. Any competition-specific rules provided by the Sponsor are in addition to these rules, and in the case of any conflict or inconsistency, these Foundational Rules control and nullify contrary competition-specific rules.

GENERAL COMPETITION RULES - BINDING AGREEMENT
1. ELIGIBILITY
a. To be eligible to enter the Competition, you must be:

a registered account holder at Kaggle.com;
the older of 18 years old or the age of majority in your jurisdiction of residence (unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian consents have been obtained by Competition Sponsor);
not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, or North Korea; and
not a person or representative of an entity under U.S. export controls or sanctions (see: https://www.treasury.gov/resourcecenter/sanctions/Programs/Pages/Programs.aspx).
b. Competitions are open to residents of the United States and worldwide, except that if you are a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, North Korea, or are subject to U.S. export controls or sanctions, you may not enter the Competition. Other local rules and regulations may apply to you, so please check your local laws to ensure that you are eligible to participate in skills-based competitions. The Competition Host reserves the right to forego or award alternative Prizes where needed to comply with local laws. If a winner is located in a country where prizes cannot be awarded, then they are not eligible to receive a prize.

c. If you are entering as a representative of a company, educational institution or other legal entity, or on behalf of your employer, these rules are binding on you, individually, and the entity you represent or where you are an employee. If you are acting within the scope of your employment, or as an agent of another party, you warrant that such party or your employer has full knowledge of your actions and has consented thereto, including your potential receipt of a Prize. You further warrant that your actions do not violate your employer's or entity's policies and procedures.

d. The Competition Sponsor reserves the right to verify eligibility and to adjudicate on any dispute at any time. If you provide any false information relating to the Competition concerning your identity, residency, mailing address, telephone number, email address, ownership of right, or information required for entering the Competition, you may be immediately disqualified from the Competition.

2. SPONSOR AND HOSTING PLATFORM
a. The Competition is sponsored by Competition Sponsor named above. The Competition is hosted on behalf of Competition Sponsor by Kaggle Inc. ("Kaggle"). Kaggle is an independent contractor of Competition Sponsor, and is not a party to this or any agreement between you and Competition Sponsor. You understand that Kaggle has no responsibility with respect to selecting the potential Competition winner(s) or awarding any Prizes. Kaggle will perform certain administrative functions relating to hosting the Competition, and you agree to abide by the provisions relating to Kaggle under these Rules. As a Kaggle.com account holder and user of the Kaggle competition platform, remember you have accepted and are subject to the Kaggle Terms of Service at www.kaggle.com/terms in addition to these Rules.

3. COMPETITION PERIOD
a. For the purposes of Prizes, the Competition will run from the Start Date and time to the Final Submission Deadline (such duration the “Competition Period”). The Competition Timeline is subject to change, and Competition Sponsor may introduce additional hurdle deadlines during the Competition Period. Any updated or additional deadlines will be publicized on the Competition Website. It is your responsibility to check the Competition Website regularly to stay informed of any deadline changes. YOU ARE RESPONSIBLE FOR DETERMINING THE CORRESPONDING TIME ZONE IN YOUR LOCATION.

4. COMPETITION ENTRY
a. NO PURCHASE NECESSARY TO ENTER OR WIN. To enter the Competition, you must register on the Competition Website prior to the Entry Deadline, and follow the instructions for developing and entering your Submission through the Competition Website. Your Submissions must be made in the manner and format, and in compliance with all other requirements, stated on the Competition Website (the "Requirements"). Submissions must be received before any Submission deadlines stated on the Competition Website. Submissions not received by the stated deadlines will not be eligible to receive a Prize. b. Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records. c. If the Competition is a multi-stage competition with temporally separate training and/or test data, one or more valid Submissions may be required during each Competition stage in the manner described on the Competition Website in order for the Submissions to be Prize eligible. d. Submissions are void if they are in whole or part illegible, incomplete, damaged, altered, counterfeit, obtained through fraud, or late. Competition Sponsor reserves the right to disqualify any entrant who does not follow these Rules, including making a Submission that does not meet the Requirements.

5. INDIVIDUALS AND TEAMS
a. Individual Account. You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy. You may submit up to the maximum number of Submissions per day as specified on the Competition Website. b. Teams. If permitted under the Competition Website guidelines, multiple individuals may collaborate as a Team; however, you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle account. You must register individually for the Competition before joining a Team. You must confirm your Team membership to make it official by responding to the Team notification message sent to your Kaggle account. Team membership may not exceed the Maximum Team Size stated on the Competition Website. c. Team Merger. Teams may request to merge via the Competition Website. Team mergers may be allowed provided that: (i) the combined Team does not exceed the Maximum Team Size; (ii) the number of Submissions made by the merging Teams does not exceed the number of Submissions permissible for one Team at the date of the merger request; (iii) the merger is completed before the earlier of: any merger deadline or the Competition deadline; and (iv) the proposed combined Team otherwise meets all the requirements of these Rules. d. Private Sharing. No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.

6. SUBMISSION CODE REQUIREMENTS
a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or Competition Specific Rules above, during the Competition Period, you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data or other source or executable code relevant to the Competition (“Competition Code”). This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs. Any such sharing of Competition Code is a breach of these Competition Rules and may result in disqualification. b. Public Code Sharing. You are permitted to publicly share Competition Code, provided that such public sharing does not violate the intellectual property rights of any third party. If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code. c. Use of Open Source. Unless otherwise stated in the Specific Competition Rules above, if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code.

7. DETERMINING WINNERS
a. Each Submission will be scored and ranked by the evaluation metric stated on the Competition Website. During the Competition Period, the current ranking will be visible on the Competition Website's Public Leaderboard. The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard, subject to compliance with these Rules. The Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set. b. In the event of a tie, the Submission that was entered first to the Competition will be the winner. In the event a potential winner is disqualified for any reason, the Submission that received the next highest score rank will be chosen as the potential winner.

8. NOTIFICATION OF WINNERS & DISQUALIFICATION
a. The potential winner(s) will be notified by email. b. If a potential winner (i) does not respond to the notification attempt within one (1) week from the first notification attempt or (ii) notifies Kaggle within one week after the Final Submission Deadline that the potential winner does not want to be nominated as a winner or does not want to receive a Prize, then, in each case (i) and (ii) such potential winner will not receive any Prize, and an alternate potential winner will be selected from among all eligible entries received based on the Competition’s judging criteria. c. In case (i) and (ii) above Kaggle may disqualify the Participant. However, in case (ii) above, if requested by Kaggle, such potential winner may provide code and documentation to verify the Participant’s compliance with these Rules. If the potential winner provides code and documentation to the satisfaction of Kaggle, the Participant will not be disqualified pursuant to this paragraph. d. Competition Sponsor reserves the right to disqualify any Participant from the Competition if the Competition Sponsor reasonably believes that the Participant has attempted to undermine the legitimate operation of the Competition by cheating, deception, or other unfair playing practices or abuses, threatens or harasses any other Participants, Competition Sponsor or Kaggle. e. A disqualified Participant may be removed from the Competition leaderboard, at Kaggle's sole discretion. If a Participant is removed from the Competition Leaderboard, additional winning features associated with the Kaggle competition platform, for example Kaggle points or medals, may also not be awarded. f. The final leaderboard list will be publicly displayed at Kaggle.com. Determinations of Competition Sponsor are final and binding.

9. PRIZES
a. Prize(s) are as described on the Competition Website and are only available for winning during the time period described on the Competition Website. The odds of winning any Prize depends on the number of eligible Submissions received during the Competition Period and the skill of the Participants. b. All Prizes are subject to Competition Sponsor's review and verification of the Participant’s eligibility and compliance with these Rules, and the compliance of the winning Submissions with the Submissions Requirements. In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions). c. A potential winner may decline to be nominated as a Competition winner in accordance with Section 3.8. d. Potential winners must return all required Prize acceptance documents within two (2) weeks following notification of such required documents, or such potential winner will be deemed to have forfeited the prize and another potential winner will be selected. Prize(s) will be awarded within approximately thirty (30) days after receipt by Competition Sponsor or Kaggle of the required Prize acceptance documents. Transfer or assignment of a Prize is not allowed. e. You are not eligible to receive any Prize if you do not meet the Eligibility requirements in Section 2.7 and Section 3.1 above. f. If a Team wins a monetary Prize, the Prize money will be allocated in even shares between the eligible Team members, unless the Team unanimously opts for a different Prize split and notifies Kaggle before Prizes are issued.

10. TAXES
a. ALL TAXES IMPOSED ON PRIZES ARE THE SOLE RESPONSIBILITY OF THE WINNERS. Payments to potential winners are subject to the express requirement that they submit all documentation requested by Competition Sponsor or Kaggle for compliance with applicable state, federal, local and foreign (including provincial) tax reporting and withholding requirements. Prizes will be net of any taxes that Competition Sponsor is required by law to withhold. If a potential winner fails to provide any required documentation or comply with applicable laws, the Prize may be forfeited and Competition Sponsor may select an alternative potential winner. Any winners who are U.S. residents will receive an IRS Form-1099 in the amount of their Prize.

11. GENERAL CONDITIONS
a. All federal, state, provincial and local laws and regulations apply.

12. PUBLICITY
a. You agree that Competition Sponsor, Kaggle and its affiliates may use your name and likeness for advertising and promotional purposes without additional compensation, unless prohibited by law.

13. PRIVACY
a. You acknowledge and agree that Competition Sponsor and Kaggle may collect, store, share and otherwise use personally identifiable information provided by you during the Kaggle account registration process and the Competition, including but not limited to, name, mailing address, phone number, and email address (“Personal Information”). Kaggle acts as an independent controller with regard to its collection, storage, sharing, and other use of this Personal Information, and will use this Personal Information in accordance with its Privacy Policy <www.kaggle.com/privacy>, including for administering the Competition. As a Kaggle.com account holder, you have the right to request access to, review, rectification, portability or deletion of any personal data held by Kaggle about you by logging into your account and/or contacting Kaggle Support at <www.kaggle.com/contact>. b. As part of Competition Sponsor performing this contract between you and the Competition Sponsor, Kaggle will transfer your Personal Information to Competition Sponsor, which acts as an independent controller with regard to this Personal Information. As a controller of such Personal Information, Competition Sponsor agrees to comply with all U.S. and foreign data protection obligations with regard to your Personal Information. Kaggle will transfer your Personal Information to Competition Sponsor in the country specified in the Competition Sponsor Address listed above, which may be a country outside the country of your residence. Such country may not have privacy laws and regulations similar to those of the country of your residence.

14. WARRANTY, INDEMNITY AND RELEASE
a. You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses. You agree not to make any Submission that: (i) infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; or (ii) otherwise violates any applicable U.S. or foreign state or federal law. b. To the maximum extent permitted by law, you indemnify and agree to keep indemnified Competition Entities at all times from and against any liability, claims, demands, losses, damages, costs and expenses resulting from any of your acts, defaults or omissions and/or a breach of any warranty set forth herein. To the maximum extent permitted by law, you agree to defend, indemnify and hold harmless the Competition Entities from and against any and all claims, actions, suits or proceedings, as well as any and all losses, liabilities, damages, costs and expenses (including reasonable attorneys fees) arising out of or accruing from: (a) your Submission or other material uploaded or otherwise provided by you that infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; (b) any misrepresentation made by you in connection with the Competition; (c) any non-compliance by you with these Rules or any applicable U.S. or foreign state or federal law; (d) claims brought by persons or entities other than the parties to these Rules arising from or related to your involvement with the Competition; and (e) your acceptance, possession, misuse or use of any Prize, or your participation in the Competition and any Competition-related activity. c. You hereby release Competition Entities from any liability associated with: (a) any malfunction or other problem with the Competition Website; (b) any error in the collection, processing, or retention of any Submission; or (c) any typographical or other error in the printing, offering or announcement of any Prize or winners.

15. INTERNET
a. Competition Entities are not responsible for any malfunction of the Competition Website or any late, lost, damaged, misdirected, incomplete, illegible, undeliverable, or destroyed Submissions or entry materials due to system errors, failed, incomplete or garbled computer or other telecommunication transmission malfunctions, hardware or software failures of any kind, lost or unavailable network connections, typographical or system/human errors and failures, technical malfunction(s) of any telephone network or lines, cable connections, satellite transmissions, servers or providers, or computer equipment, traffic congestion on the Internet or at the Competition Website, or any combination thereof, which may limit a Participant’s ability to participate.

16. RIGHT TO CANCEL, MODIFY OR DISQUALIFY
a. If for any reason the Competition is not capable of running as planned, including infection by computer virus, bugs, tampering, unauthorized intervention, fraud, technical failures, or any other causes which corrupt or affect the administration, security, fairness, integrity, or proper conduct of the Competition, Competition Sponsor reserves the right to cancel, terminate, modify or suspend the Competition. Competition Sponsor further reserves the right to disqualify any Participant who tampers with the submission process or any other part of the Competition or Competition Website. Any attempt by a Participant to deliberately damage any website, including the Competition Website, or undermine the legitimate operation of the Competition is a violation of criminal and civil laws. Should such an attempt be made, Competition Sponsor and Kaggle each reserves the right to seek damages from any such Participant to the fullest extent of the applicable law.

17. NOT AN OFFER OR CONTRACT OF EMPLOYMENT
a. Under no circumstances will the entry of a Submission, the awarding of a Prize, or anything in these Rules be construed as an offer or contract of employment with Competition Sponsor or any of the Competition Entities. You acknowledge that you have submitted your Submission voluntarily and not in confidence or in trust. You acknowledge that no confidential, fiduciary, agency, employment or other similar relationship is created between you and Competition Sponsor or any of the Competition Entities by your acceptance of these Rules or your entry of your Submission.

18. DEFINITIONS
a. "Competition Data" are the data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to Participants. b. An “Entry” is when a Participant has joined, signed up, or accepted the rules of a competition. Entry is required to make a Submission to a competition. c. A “Final Submission” is the Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard. d. A “Participant” or “Participant User” is an individual who participates in a competition by entering the competition and making a Submission. e. The “Private Leaderboard” is a ranked display of Participants’ Submission scores against the private test set. The Private Leaderboard determines the final standing in the competition. f. The “Public Leaderboard” is a ranked display of Participants’ Submission scores against a representative sample of the test data. This leaderboard is visible throughout the competition. g. A “Sponsor” is responsible for hosting the competition, which includes but is not limited to providing the data for the competition, determining winners, and enforcing competition rules. h. A “Submission” is anything provided by the Participant to the Sponsor to be evaluated for competition purposes and determine leaderboard position. A Submission may be made as a model, notebook, prediction file, or other format as determined by the Sponsor. i. A “Team” is one or more Participants participating together in a Kaggle competition, by officially merging together as a Team within the competition platform.

Rules


